#!/usr/bin/env bash
set -euo pipefail

API="https://truthlens-ispk.onrender.com"
ADMIN_KEY="hackathon-admin-key"        # must match Render env
SECRET="hackathon-secret-123"          # must match Render env

echo "→ warmup"
curl -s "$API/warmup" | jq .

echo "→ create job"
RESP=$(curl -s -X POST "$API/analyze" -F "file=@deepfake_detection_agent/samples/kiss.mp4;type=video/mp4")
echo "$RESP" | jq .

JOB=$(echo "$RESP" | jq -r '.job_id')
echo "JOB=$JOB"

echo "→ check server has this job"
J=$(curl -s "$API/admin/jobs" -H "x-admin-key: $ADMIN_KEY" | jq ".\"$JOB\"")
echo "$J"

if [[ "$J" == "null" ]]; then
  echo "…not visible yet, sleeping 3s"
  sleep 3
  J=$(curl -s "$API/admin/jobs" -H "x-admin-key: $ADMIN_KEY" | jq ".\"$JOB\"")
  echo "$J"
  [[ "$J" == "null" ]] && { echo "Job still not found on server; aborting."; exit 1; }
fi

echo "→ compute signature"
SIG=$(printf '%s' "$JOB" | openssl dgst -sha256 -hmac "$SECRET" | sed 's/^.* //')
echo "SIG=$SIG"

echo "→ approve"
curl -sS -i "$API/jobs/$JOB/approve?sig=$SIG"

echo "→ fetch approved job"
curl -s "$API/jobs/$JOB" | jq .
