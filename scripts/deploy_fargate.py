"""Build the image, deploy to Fargate, verify, scale to zero. All boto3.

    python -m scripts.deploy_fargate build          # zip -> S3 -> CodeBuild -> ECR
    python -m scripts.deploy_fargate build --local  # docker build here -> ECR (no CodeBuild quota needed)
    python -m scripts.deploy_fargate secrets    # .env secrets -> SSM SecureString
    python -m scripts.deploy_fargate deploy     # roles, cluster, task def, service
    python -m scripts.deploy_fargate verify     # /healthz, /readyz, one /query
    python -m scripts.deploy_fargate scale 0    # stop paying; `scale 1` to resume
    python -m scripts.deploy_fargate teardown   # delete service + cluster (keeps ECR/SSM)

Every step is idempotent: re-running reuses what exists. Names are prefixed
`rag-`. No Terraform/CDK by design - this is the committed record of what the
deployment consists of. The CodeBuild path needs no Docker on the developer
machine; `--local` is the escape hatch for accounts whose CodeBuild
concurrency quota is 0 (this one, at the time of writing).
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from core.config import Settings, get_settings

PREFIX = "rag"
ECR_REPO = f"{PREFIX}-api"
CLUSTER = f"{PREFIX}-cluster"
SERVICE = f"{PREFIX}-api"
TASK_FAMILY = f"{PREFIX}-api"
CONTAINER = "api"
PORT = 8000
LOG_GROUP = f"/ecs/{PREFIX}-api"
CODEBUILD_PROJECT = f"{PREFIX}-image-build"
CODEBUILD_ROLE = f"{PREFIX}-codebuild-role"
EXEC_ROLE = f"{PREFIX}-ecs-execution-role"
SSM_PREFIX = f"/{PREFIX}/"
CPU, MEMORY = "512", "1024"

# Secrets pulled from SSM at task start; everything else is plain environment.
SECRET_KEYS = [
    "PORTKEY_API_KEY", "PORTKEY_CONFIG_SLUG", "GROQ_API_KEY", "COHERE_API_KEY",
    "QDRANT_URL", "QDRANT_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
]
# Non-secret config mirrored from the local (measured) configuration.
CONFIG_KEYS = [
    "LLM_PRIMARY_PROVIDER", "LLM_PRIMARY_MODEL", "GROQ_MODEL_ID", "GROQ_REASONING_EFFORT",
    "EMBEDDING_BACKEND", "COHERE_EMBED_MODEL", "COHERE_EMBED_DIM", "COHERE_EMBED_TPM",
    "RERANKER_BACKEND", "COHERE_RERANK_MODEL", "RETRIEVAL_MODE", "PREFETCH_K",
    "CHUNK_STRATEGY", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "RERANK_TOP_N",
    "QDRANT_COLLECTION_PREFIX", "OUTPUT_GUARD_BACKEND", "INPUT_GUARD_ENGINE",
    "INJECTION_BACKEND", "INJECTION_MODEL_ID", "LANGFUSE_HOST", "OTEL_SERVICE_NAME",
    "AWS_REGION",
]
EXCLUDE_DIRS = {".git", "data", "tests", "docs", "results", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv", "venv"}
EXCLUDE_FILES = {".env", "CLAUDE.md", "AI_Engineer_Portfolio_Projects.md"}


def _session(settings: Settings) -> boto3.Session:
    return boto3.Session(
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
    )


def _account(session: boto3.Session) -> str:
    return session.client("sts").get_caller_identity()["Account"]


def say(msg: str) -> None:
    print(msg, flush=True)


# ---- IAM ------------------------------------------------------------------------


def ensure_role(iam: Any, name: str, service: str, managed: list[str], inline: dict[str, Any] | None) -> str:
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]}
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust), Description=f"{PREFIX} {service}")["Role"]["Arn"]
        say(f"  created role {name}; waiting for IAM propagation")
        time.sleep(12)
    for policy in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=policy)
    if inline:
        iam.put_role_policy(RoleName=name, PolicyName=f"{name}-inline", PolicyDocument=json.dumps(inline))
    return arn


# ---- build ------------------------------------------------------------------------


def zip_source(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if path.is_file() and rel.name not in EXCLUDE_FILES and not rel.name.startswith(".env."):
                zf.write(path, rel.as_posix())
    return buf.getvalue()


def ensure_ecr_repo(ecr: Any) -> None:
    try:
        ecr.create_repository(repositoryName=ECR_REPO, imageScanningConfiguration={"scanOnPush": True})
        say(f"created ECR repo {ECR_REPO}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "RepositoryAlreadyExistsException":
            raise


def _docker(*argv: str, input: bytes | None = None) -> None:
    say("  $ docker " + " ".join(a if len(a) < 80 else a[:77] + "..." for a in argv))
    subprocess.run(["docker", *argv], check=True, input=input)


def cmd_build_local(settings: Settings, args: argparse.Namespace) -> int:
    """docker build on this machine, push to ECR. Same image, same tag."""
    session = _session(settings)
    account = _account(session)
    region = settings.aws_region
    ecr = session.client("ecr")
    ensure_ecr_repo(ecr)
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"
    image = f"{registry}/{ECR_REPO}:{args.tag}"

    auth = ecr.get_authorization_token()["authorizationData"][0]
    import base64

    password = base64.b64decode(auth["authorizationToken"]).split(b":", 1)[1]
    # The token never touches the command line or the log: docker reads it from stdin.
    _docker("login", "--username", "AWS", "--password-stdin", registry, input=password)

    started = time.perf_counter()
    _docker("build", "--platform", "linux/amd64", "-f", "docker/Dockerfile", "-t", image, ".")
    say(f"built in {time.perf_counter() - started:.0f}s")
    size = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Size}}"], check=True, capture_output=True, text=True
    ).stdout.strip()
    say(f"image size: {int(size) / 1e6:.0f} MB (uncompressed)")
    _docker("push", image)
    digest = ecr.describe_images(repositoryName=ECR_REPO, imageIds=[{"imageTag": args.tag}])["imageDetails"][0]
    say(f"pushed {image}  digest={digest['imageDigest'][:19]}  "
        f"compressed={digest['imageSizeInBytes'] / 1e6:.0f} MB")
    return 0


def cmd_build(settings: Settings, args: argparse.Namespace) -> int:
    if args.local:
        return cmd_build_local(settings, args)
    session = _session(settings)
    account = _account(session)
    region = settings.aws_region
    ecr, s3, cb, iam = (session.client(x) for x in ("ecr", "s3", "codebuild", "iam"))
    ensure_ecr_repo(ecr)
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"
    image_repo = f"{registry}/{ECR_REPO}"

    bucket = f"{PREFIX}-build-{account}-{region}"
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        kwargs: dict[str, Any] = {"Bucket": bucket}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)
        s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        say(f"created S3 bucket {bucket}")

    payload = zip_source(Path(".").resolve())
    s3.put_object(Bucket=bucket, Key="source.zip", Body=payload)
    say(f"uploaded source.zip ({len(payload) // 1024} KB)")

    role_arn = ensure_role(
        iam, CODEBUILD_ROLE, "codebuild.amazonaws.com", [],
        {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion"], "Resource": f"arn:aws:s3:::{bucket}/*"},
            {"Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["ecr:BatchCheckLayerAvailability", "ecr:CompleteLayerUpload", "ecr:InitiateLayerUpload",
                                            "ecr:PutImage", "ecr:UploadLayerPart", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
             "Resource": f"arn:aws:ecr:{region}:{account}:repository/{ECR_REPO}"},
        ]},
    )

    project = {
        "name": CODEBUILD_PROJECT,
        "source": {"type": "S3", "location": f"{bucket}/source.zip", "buildspec": "docker/buildspec.yml"},
        "artifacts": {"type": "NO_ARTIFACTS"},
        "environment": {
            "type": "LINUX_CONTAINER", "image": "aws/codebuild/standard:7.0", "computeType": "BUILD_GENERAL1_SMALL",
            "privilegedMode": True,
            "environmentVariables": [
                {"name": "AWS_REGION", "value": region}, {"name": "ECR_REGISTRY", "value": registry},
                {"name": "ECR_REPO", "value": image_repo}, {"name": "IMAGE_TAG", "value": args.tag},
            ],
        },
        "serviceRole": role_arn,
        "timeoutInMinutes": 30,
    }
    try:
        cb.create_project(**project)
        say(f"created CodeBuild project {CODEBUILD_PROJECT}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise
        cb.update_project(**project)

    build_id = cb.start_build(projectName=CODEBUILD_PROJECT)["build"]["id"]
    say(f"build started: {build_id}")
    phase = None
    while True:
        b = cb.batch_get_builds(ids=[build_id])["builds"][0]
        if b.get("currentPhase") != phase:
            phase = b.get("currentPhase")
            say(f"  phase: {phase}")
        if b["buildStatus"] != "IN_PROGRESS":
            break
        time.sleep(15)
    say(f"build {b['buildStatus']} in {(b.get('endTime', b['startTime']) - b['startTime']).total_seconds():.0f}s")
    logs = b.get("logs", {})
    if logs.get("groupName"):
        cw = session.client("logs")
        try:
            events = cw.get_log_events(logGroupName=logs["groupName"], logStreamName=logs["streamName"], limit=40)["events"]
            tail = [e["message"].rstrip() for e in events][-25:]
            say("  --- build log tail ---")
            for line in tail:
                if line.strip():
                    say("  " + line[:160])
        except ClientError:
            pass
    if b["buildStatus"] != "SUCCEEDED":
        return 1
    say(f"image: {image_repo}:{args.tag}")
    return 0


# ---- secrets ----------------------------------------------------------------------


def cmd_secrets(settings: Settings, args: argparse.Namespace) -> int:
    from dotenv import dotenv_values

    env = dotenv_values(".env")
    ssm = _session(settings).client("ssm")
    for key in SECRET_KEYS:
        value = env.get(key)
        if not value:
            say(f"  skip {key} (not in .env)")
            continue
        ssm.put_parameter(Name=f"{SSM_PREFIX}{key}", Value=value, Type="SecureString", Overwrite=True)
        say(f"  put {SSM_PREFIX}{key}")
    return 0


# ---- deploy -----------------------------------------------------------------------


def _default_network(ec2: Any) -> tuple[str, list[str]]:
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    subnets = [s["SubnetId"] for s in ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]}])["Subnets"]]
    return vpc, subnets


def _ensure_sg(ec2: Any, vpc: str) -> str:
    name = f"{PREFIX}-api-sg"
    found = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [name]}, {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    if found:
        return found[0]["GroupId"]
    sg = ec2.create_security_group(GroupName=name, Description=f"{PREFIX} api ingress", VpcId=vpc)["GroupId"]
    ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
        "IpProtocol": "tcp", "FromPort": PORT, "ToPort": PORT, "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "demo"}]}])
    say(f"  created security group {name}")
    return sg


def cmd_deploy(settings: Settings, args: argparse.Namespace) -> int:
    from dotenv import dotenv_values

    session = _session(settings)
    account = _account(session)
    region = settings.aws_region
    ecs, ec2, iam, logs, ssm = (session.client(x) for x in ("ecs", "ec2", "iam", "logs", "ssm"))
    image = f"{account}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO}:{args.tag}"

    try:
        logs.create_log_group(logGroupName=LOG_GROUP)
        logs.put_retention_policy(logGroupName=LOG_GROUP, retentionInDays=14)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise

    exec_role = ensure_role(
        iam, EXEC_ROLE, "ecs-tasks.amazonaws.com",
        ["arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"],
        {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["ssm:GetParameters", "ssm:GetParameter"],
             "Resource": f"arn:aws:ssm:{region}:{account}:parameter{SSM_PREFIX}*"},
            {"Effect": "Allow", "Action": ["kms:Decrypt"], "Resource": "*"},
        ]},
    )

    # A fresh account has no ECS service-linked role until something creates
    # it; CreateCluster then fails with "Unable to assume the service linked role".
    try:
        iam.create_service_linked_role(AWSServiceName="ecs.amazonaws.com")
        say("  created service-linked role AWSServiceRoleForECS; waiting for IAM propagation")
        time.sleep(12)
    except ClientError as e:
        if e.response["Error"]["Code"] != "InvalidInput":  # already exists
            raise

    try:
        ecs.create_cluster(clusterName=CLUSTER, capacityProviders=["FARGATE"])
        say(f"  cluster {CLUSTER} ready")  # CreateCluster is idempotent by name
    except ClientError as e:
        if "already exists" not in str(e):
            raise

    env_file = dotenv_values(".env")
    environment = [{"name": k, "value": str(env_file.get(k) or _default_config(settings, k))} for k in CONFIG_KEYS
                   if env_file.get(k) or _default_config(settings, k) is not None]
    environment.append({"name": "LANGFUSE_ENABLED", "value": "true"})
    present = {p["Name"].removeprefix(SSM_PREFIX) for p in ssm.get_parameters_by_path(Path=SSM_PREFIX)["Parameters"]}
    secrets = [{"name": k, "valueFrom": f"arn:aws:ssm:{region}:{account}:parameter{SSM_PREFIX}{k}"} for k in SECRET_KEYS if k in present]
    missing = [k for k in SECRET_KEYS if k not in present]
    if missing:
        say(f"  WARNING secrets not in SSM (run `secrets` first): {missing}")

    td = ecs.register_task_definition(
        family=TASK_FAMILY, networkMode="awsvpc", requiresCompatibilities=["FARGATE"],
        cpu=CPU, memory=MEMORY, executionRoleArn=exec_role,
        runtimePlatform={"cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX"},
        containerDefinitions=[{
            "name": CONTAINER, "image": image, "essential": True,
            "portMappings": [{"containerPort": PORT, "protocol": "tcp"}],
            "environment": environment, "secrets": secrets,
            "logConfiguration": {"logDriver": "awslogs", "options": {
                "awslogs-group": LOG_GROUP, "awslogs-region": region, "awslogs-stream-prefix": "api"}},
            "healthCheck": {"command": ["CMD-SHELL", f"curl -fsS http://127.0.0.1:{PORT}/healthz || exit 1"],
                            "interval": 30, "timeout": 5, "retries": 3, "startPeriod": 60},
        }],
    )["taskDefinition"]
    td_arn = td["taskDefinitionArn"]
    say(f"  registered task definition {td['family']}:{td['revision']}")

    vpc, subnets = _default_network(ec2)
    sg = _ensure_sg(ec2, vpc)
    network = {"awsvpcConfiguration": {"subnets": subnets, "securityGroups": [sg], "assignPublicIp": "ENABLED"}}

    existing = ecs.describe_services(cluster=CLUSTER, services=[SERVICE])["services"]
    if existing and existing[0]["status"] == "ACTIVE":
        ecs.update_service(cluster=CLUSTER, service=SERVICE, taskDefinition=td_arn, desiredCount=1, forceNewDeployment=True)
        say(f"  updated service {SERVICE}")
    else:
        ecs.create_service(cluster=CLUSTER, serviceName=SERVICE, taskDefinition=td_arn, desiredCount=1,
                           launchType="FARGATE", networkConfiguration=network)
        say(f"  created service {SERVICE}")

    say("  waiting for the task to reach RUNNING ...")
    ip = None
    for _ in range(40):
        time.sleep(15)
        ip, status = _task_ip(session)
        say(f"    {status}")
        if status == "RUNNING" and ip:
            break
    if not ip:
        say("  task did not reach RUNNING; check CloudWatch " + LOG_GROUP)
        return 1
    say(f"public ip: {ip}  ->  http://{ip}:{PORT}/healthz")
    return 0


def _default_config(settings: Settings, key: str) -> Any:
    value = getattr(settings, key.lower(), None)
    if value is None:
        return None
    return value.value if hasattr(value, "value") else value


def _task_ip(session: boto3.Session) -> tuple[str | None, str]:
    ecs, ec2 = session.client("ecs"), session.client("ec2")
    arns = ecs.list_tasks(cluster=CLUSTER, serviceName=SERVICE, desiredStatus="RUNNING")["taskArns"]
    if not arns:
        arns = ecs.list_tasks(cluster=CLUSTER, serviceName=SERVICE)["taskArns"]
    if not arns:
        return None, "no tasks"
    task = ecs.describe_tasks(cluster=CLUSTER, tasks=arns[:1])["tasks"][0]
    status = task["lastStatus"]
    if status != "RUNNING":
        reason = task.get("stoppedReason") or (task["containers"][0].get("reason") if task.get("containers") else "")
        return None, f"{status} {reason or ''}".strip()
    eni = next((d["value"] for a in task["attachments"] for d in a["details"] if d["name"] == "networkInterfaceId"), None)
    if not eni:
        return None, "RUNNING (no eni yet)"
    ip = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni])["NetworkInterfaces"][0].get("Association", {}).get("PublicIp")
    return ip, "RUNNING"


# ---- verify / scale / teardown ---------------------------------------------------


def cmd_verify(settings: Settings, args: argparse.Namespace) -> int:
    import httpx

    session = _session(settings)
    ip, status = _task_ip(session)
    if not ip:
        say(f"no running task ({status})")
        return 1
    base = f"http://{ip}:{PORT}"
    with httpx.Client(timeout=120) as c:
        # Cold start: from the task's RUNNING timestamp to the first 200 from
        # /healthz. Polled, so `verify` right after `deploy` measures it
        # instead of failing on a connection refused during startup.
        started_at = _task_started_at(session)
        deadline = time.time() + 180
        while True:
            try:
                r = c.get(base + "/healthz", timeout=5)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                say("app did not become healthy within 180s")
                return 1
            time.sleep(3)
        if started_at:
            say(f"cold start: {time.time() - started_at.timestamp():.0f}s from task RUNNING to first healthy response")
        for path in ("/healthz", "/readyz"):
            r = c.get(base + path)
            say(f"GET {path} -> {r.status_code} {r.text[:160]}")
        question = args.question or "Which five new districts have been formed in the Union Territory of Ladakh?"
        t0 = time.perf_counter()
        r = c.post(base + "/query", json={"question": question})
        say(f"POST /query -> {r.status_code} in {time.perf_counter() - t0:.1f}s")
        body = r.json()
        say(json.dumps({k: body.get(k) for k in ("status", "answer", "grounded", "latency_ms", "config_fingerprint")}, indent=1, ensure_ascii=False))
        say(f"usage: {body.get('usage')}")
        say(f"citations: {[(c_['circular_no'], c_['page']) for c_ in body.get('citations', [])]}")
    return 0


def _task_started_at(session: boto3.Session) -> Any:
    ecs = session.client("ecs")
    arns = ecs.list_tasks(cluster=CLUSTER, serviceName=SERVICE, desiredStatus="RUNNING")["taskArns"]
    if not arns:
        return None
    return ecs.describe_tasks(cluster=CLUSTER, tasks=arns[:1])["tasks"][0].get("startedAt")


def cmd_scale(settings: Settings, args: argparse.Namespace) -> int:
    ecs = _session(settings).client("ecs")
    ecs.update_service(cluster=CLUSTER, service=SERVICE, desiredCount=args.count)
    say(f"desired count -> {args.count}")
    return 0


def cmd_teardown(settings: Settings, args: argparse.Namespace) -> int:
    ecs = _session(settings).client("ecs")
    try:
        ecs.update_service(cluster=CLUSTER, service=SERVICE, desiredCount=0)
        ecs.delete_service(cluster=CLUSTER, service=SERVICE, force=True)
        say(f"deleted service {SERVICE}")
    except ClientError as e:
        say(f"service: {e.response['Error']['Code']}")
    try:
        ecs.delete_cluster(cluster=CLUSTER)
        say(f"deleted cluster {CLUSTER}")
    except ClientError as e:
        say(f"cluster: {e.response['Error']['Code']}")
    say("kept: ECR repo, SSM parameters, IAM roles, log group, S3 build bucket")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--tag", default="latest")
    b.add_argument("--local", action="store_true", help="docker build on this machine instead of CodeBuild")
    sub.add_parser("secrets")
    d = sub.add_parser("deploy"); d.add_argument("--tag", default="latest")
    v = sub.add_parser("verify"); v.add_argument("--question", default=None)
    s = sub.add_parser("scale"); s.add_argument("count", type=int)
    sub.add_parser("teardown")
    args = p.parse_args()
    settings = get_settings()
    return {"build": cmd_build, "secrets": cmd_secrets, "deploy": cmd_deploy,
            "verify": cmd_verify, "scale": cmd_scale, "teardown": cmd_teardown}[args.cmd](settings, args)


if __name__ == "__main__":
    sys.exit(main())
