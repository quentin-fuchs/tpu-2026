export TPU_NAME=boris-2
export PROJECT_ID=tpu-2026
export ZONE=us-east5-b
export ACCELERATOR_TYPE=v6e-1
export VERSION=v2-alpha-tpuv6e

# One-time per region: enable Private Google Access on the default subnet
# so internal-IP TPUs can reach Google APIs.
gcloud compute networks subnets update default \
  --region=us-east5 \
  --enable-private-ip-google-access \
  --project=$PROJECT_ID

gcloud compute tpus tpu-vm create $TPU_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --accelerator-type=$ACCELERATOR_TYPE \
  --version=$VERSION \
  --internal-ips
