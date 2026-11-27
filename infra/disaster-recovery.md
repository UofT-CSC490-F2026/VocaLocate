# Disaster recovery runbook (A2 Part Five)

The script for the recorded demo: prod is destroyed and then rebuilt from code, with its data
restored from the backup stack. Every command runs from the repository root in Git Bash, signed in
to the team account (266727734425). Each section says what to show on screen; together they cover
what Part Five asks for: services, databases and data, configuration, access controls and
verification.

## What survives a prod teardown, and why

| Piece | Where it lives | After `terraform destroy` of prod |
| --- | --- | --- |
| Services (VPC, ECS, ECR, Step Functions, EventBridge, Glue, Athena, alarms) | `infra/envs/prod` state | Gone; `terraform apply` recreates them from the same code |
| Data (raw, lake incl. the DuckDB warehouse, models) | prod buckets, replicated to `vocalocate-backup-*` in us-east-2 | Prod buckets gone; the backup copies stay, because the backup has its own state and deletes are not replicated |
| Configuration | Terraform variables, `pipeline/configs/prod.toml` baked into the image, the task definition's environment | Recreated by `apply` and the image push |
| Access controls | IAM roles and policies in the prod state; OIDC providers in `bootstrap` | Roles recreated with the same names and ARNs, so the GitHub `prod` environment and the Modal secret keep working unchanged |
| Pipeline image | prod's ECR repository | Gone with the repository; the `pipeline-image` workflow pushes the same commit again |

## 0. Before recording

```bash
eval "$(aws configure export-credentials --profile tanay --format env)"   # Terraform cannot read `aws login` sessions
export MSYS_NO_PATHCONV=1   # stops Git Bash rewriting /vocalocate/... log group names into Windows paths
aws sts get-caller-identity --query Account --output text                  # 266727734425
cd infra/envs/prod
SHA=$(aws ecs describe-task-definition --task-definition vocalocate-prod-pipeline \
  --query 'taskDefinition.containerDefinitions[0].image' --output text | cut -d: -f2)   # commit prod runs
ACCOUNT=266727734425
SFN=$(terraform output -raw state_machine_arn)
```

- Prod has data, and it has reached the backup. Each count pair should match:

  ```bash
  for k in raw lake models; do
    echo "$k prod=$(aws s3 ls s3://vocalocate-prod-$k-$ACCOUNT --recursive | wc -l)" \
         "backup=$(aws s3 ls s3://vocalocate-backup-$k-$ACCOUNT --recursive --region us-east-2 | wc -l)"
  done
  ```

- Note what to compare after the restore: the table counts in the last report
  (`aws logs tail /vocalocate/prod/pipeline --since 7d`) and the warehouse file
  (`aws s3 ls s3://vocalocate-prod-lake-$ACCOUNT/warehouse/`).
- Allow destroying non-empty buckets for the demo (prod refuses by default):
  `terraform apply -var image_tag=$SHA -var force_destroy=true`. This only changes the buckets'
  and the repository's `force_destroy` settings.

## 1. The disaster (on screen)

```bash
terraform destroy -var image_tag=$SHA -var force_destroy=true
```

Show: the plan's destroy count, then in the console that the prod buckets, the ECS cluster, the
state machine and the ECR repository are gone, and that `vocalocate-backup-*` in us-east-2 still
holds the data (`aws s3 ls s3://vocalocate-backup-lake-$ACCOUNT/ --region us-east-2`).

## 2. Restore the services (services, configuration, access controls)

```bash
terraform apply -var image_tag=$(git rev-parse HEAD)   # see below
```

The `pipeline-image` workflow can only build the head of a branch, so if `main` has moved since
prod was deployed, rebuild with the current `main` commit (as above) rather than `$SHA`, and push
that commit's image. Check first that nothing under `pipeline/` changed in between
(`git diff --stat $SHA HEAD -- pipeline`), so the restored environment runs the same code.

Show: the plan only adds (the same resources as before), and the outputs. Point out:

- **Configuration:** the task definition again carries `VOCALOCATE_CONFIG=/app/configs/prod.toml`
  and the bucket URIs (`aws ecs describe-task-definition --task-definition vocalocate-prod-pipeline
  --query 'taskDefinition.containerDefinitions[0].environment'`).
- **Access controls:** `aws iam get-role --role-name vocalocate-prod-github-actions --query
  'Role.AssumeRolePolicyDocument.Statement[0].Condition'` shows the same OIDC trust, and the role
  ARN is unchanged, so nothing in GitHub or Modal needs updating. Bucket policies again deny
  non-TLS access and the public access block is on.
- Then push the image again: GitHub → Actions → **pipeline-image** → Run workflow, branch `main`,
  environment **`prod`** (the field defaults to `dev`). Wait for the green `push` job and check the
  tag arrived: `aws ecr describe-images --repository-name vocalocate-prod-pipeline --query
  'imageDetails[].imageTags'`.
- The SNS alert topic was recreated: confirm the new subscription email, or prod alerts stop.

## 3. Restore the data (databases and data)

The DuckDB warehouse is a file in the lake bucket (`warehouse/vocalocate.duckdb`), so restoring the
lake restores the database; the Glue catalog is rebuilt from the lake in step 4.

```bash
for k in raw lake models; do
  aws s3 sync s3://vocalocate-backup-$k-$ACCOUNT s3://vocalocate-prod-$k-$ACCOUNT \
    --source-region us-east-2 --region ca-central-1
done
```

Show: the same object counts as before the disaster (the loop from step 0).

## 4. Verify end to end

```bash
aws stepfunctions start-execution --state-machine-arn "$SFN" --input '{}'
```

With no source, the state machine skips ingest and runs LoadWarehouse → Report → Catalog: it
rebuilds the warehouse from the restored tables, runs the data-quality report and re-registers the
tables in Glue. Show:

- the execution succeeding in the Step Functions console (each state green);
- the report in the ECS task's log (CloudWatch log group `/vocalocate/prod/pipeline`), with the same
  counts as before the disaster;
- the Glue tables (`aws glue get-tables --database-name vocalocate_prod_lake --query
  'TableList[].Name'`), and optionally one Athena query in workgroup `vocalocate-prod`. The
  Catalog step only starts the crawler, so wait until `aws glue get-crawler --name
  vocalocate-prod-lake --query Crawler.State` is `READY` before listing the tables.

Optionally, upload one file to `raw/` to show the upload trigger and replication working again
on the rebuilt environment.

## 5. After recording

Set prod back to refusing to destroy non-empty buckets: `terraform apply -var image_tag=$SHA`
(already the case if step 2 ran without `force_destroy`).
