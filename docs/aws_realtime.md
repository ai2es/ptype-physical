# AWS access for the realtime jobs

Realtime jobs run unattended every hour, so they cannot use a single sign-on profile: its login expires
(that stopped the archive generation once) and nobody is there to renew it. They need a long-lived key that
can do only what the pipeline does, in only the part of the bucket it writes.

**Who does what.** You create the IAM user and its key and put the key in a profile on glade. The pipeline is
given only the *name* of that profile. Nothing in this repository, its job scripts or its logs contains the key.

## 1. Policy

Create an IAM policy with this document (replace nothing; the bucket name is fixed):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListOnlyThePtypeAndSoundingPrefixes",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::wwe-ptype",
      "Condition": { "StringLike": { "s3:prefix": ["ptype/*", "soundings/*"] } }
    },
    {
      "Sid": "ReadAndWriteStoresAndTheIndex",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
      "Resource": ["arn:aws:s3:::wwe-ptype/ptype/*", "arn:aws:s3:::wwe-ptype/soundings/*"]
    },
    {
      "Sid": "DeleteRealtimeStagingStores",
      "Effect": "Allow",
      "Action": "s3:DeleteObject",
      "Resource": "arn:aws:s3:::wwe-ptype/ptype/realtime-staging/*"
    }
  ]
}
```

What each statement is for:

| Statement | Needed by |
|---|---|
| List, limited to `ptype/` and `soundings/` | skipping objects already uploaded; finding old staging stores |
| Get, Put, AbortMultipartUpload | uploading stores; reading and conditionally rewriting `ptype/inits.json`; writing `ptype/realtime_status.json` |
| Delete under `ptype/realtime-staging/` only | removing a run's per-hour staging store after its consolidated store is published |

The key cannot delete archive stores, soundings or the index, change the bucket policy, or touch any other
bucket.

**Retention is separate.** `applications/hrrr_ptype_expire.py --delete` removes whole runs, which needs
`s3:DeleteObject` on `arn:aws:s3:::wwe-ptype/ptype/*` and `arn:aws:s3:::wwe-ptype/soundings/*`. Leave that
out of this key unless you decide to run retention unattended; run it by hand with your own login instead.

## 2. User and key

1. Create an IAM user for programmatic access only (no console password), for example `wwe-ptype-realtime`,
   and attach the policy above and nothing else.
2. Create one access key for it.
3. On glade, add a profile to `~/.aws/credentials` (the file must stay mode 600):

   ```ini
   [wwe-ptype-realtime]
   aws_access_key_id = ...
   aws_secret_access_key = ...
   ```

   and to `~/.aws/config`:

   ```ini
   [profile wwe-ptype-realtime]
   region = us-east-1
   ```

4. Check it from a login node. The first command must succeed and the second must be refused:

   ```bash
   AWS_PROFILE=wwe-ptype-realtime aws s3api head-object --bucket wwe-ptype --key ptype/inits.json
   ```

   ```bash
   AWS_PROFILE=wwe-ptype-realtime aws s3 ls s3://wwe-ptype/
   ```

   `head-object` is the test to trust. `aws sts get-caller-identity` kept succeeding after the single sign-on
   login had expired for S3.

5. Put the profile name (not the key) in `~/.ptype_realtime.env`, see `docs/realtime.md`.

Rotate the key on whatever schedule your account requires: create a second key, replace the two lines in
`~/.aws/credentials`, delete the old key. Jobs pick up the new one at their next start.
