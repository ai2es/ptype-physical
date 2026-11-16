# Publishing HRRR precipitation-type output to S3 (`wwe-ptype`)

This is the procedure for getting a run's output from Derecho into the bucket
the website reads, and for configuring that bucket. Commands in sections 2 and 3
change AWS state: read them, then run them yourself. Statements marked
**[docs]** come from AWS documentation and were not tested while this was written.
Everything else was observed (the Derecho credential setup, the `obstore` API) or
is a plain description of files in this repo.

Bucket layout:

```
s3://wwe-ptype/ptype/inits.json
s3://wwe-ptype/ptype/YYYY/MM/DD/hrrr_ptype_YYYYMMDD_HH00.zarr/...
s3://wwe-ptype/soundings/YYYY/MM/DD/hrrr_soundings_YYYYMMDD_HH00.zarr/...
```

## 1. Credentials on Derecho

### What the profile is

Inspecting `~/.aws/config` and `~/.aws/credentials` handling on Derecho (AWS CLI 2.2.19):

- `ai2es-testbed-user` is **not** defined in `~/.aws/config`. `aws configure list --profile ai2es-testbed-user`
  shows its access key and secret key come from the **shared credentials file** (`~/.aws/credentials`): static-looking
  keys that no AWS tool refreshes. There is no `sso_*`, `role_arn`/`source_profile` or `credential_process` entry for it.
- Calling AWS with it currently fails with `ExpiredToken`. An expired *token* means the keys in the file are
  **temporary** (an access key + secret + session token issued by STS), pasted in at some earlier time. They stop working
  when the issuer's session ends, and nothing renews them.
- `~/.aws/config` does contain a separate IAM Identity Center (SSO) profile for an administrator role
  (`sso_start_url`, `sso_region`, `sso_account_id`, `sso_role_name`). It is not linked to `ai2es-testbed-user`.
  Whether it is the same AWS account as `wwe-ptype` was not checked.

I could not see how the temporary keys were originally obtained, and did not read the credentials file.

### Refreshing the keys

Because the profile is just keys in a file, logging in means getting new temporary keys and putting them in that file:

1. Sign in to the AWS access portal (the SSO start URL for your organization) in a browser on your laptop, choose the account and role for
   the testbed user, and open "Command line or programmatic access". **[docs]** This page offers the current
   `aws_access_key_id`, `aws_secret_access_key` and `aws_session_token`. (Ask whoever administers the account if the
   keys come from somewhere else.)
2. On Derecho, paste them under the `[ai2es-testbed-user]` header in `~/.aws/credentials` (all three lines; the session token
   line is what makes them temporary). Edit the file in your editor; do not paste keys into shell commands, which land in
   shell history. `chmod 600 ~/.aws/credentials`.
3. Alternative if the SSO profile is for the right account: `aws sso login --profile <that-profile>` and then use that
   profile name instead of `ai2es-testbed-user` everywhere below. On a headless login node the CLI cannot open a browser;
   `--no-browser` prints a URL and code to open on your laptop. **[docs]** The `--use-device-code` flag is newer than the
   CLI 2.2.19 installed on Derecho, so expect it not to exist there; `aws sso login help` will tell you.
   A separate newer AWS CLI (e.g. in a conda env) is another way out.

Check it worked (the first command is what failed before):

```bash
export AWS_PROFILE=ai2es-testbed-user
aws sts get-caller-identity                    # prints Account / Arn instead of ExpiredToken
aws s3 ls s3://wwe-ptype/ptype/                # lists the prefix (needs ListBucket for YOUR identity)
aws s3api get-bucket-location --bucket wwe-ptype   # "LocationConstraint": null means us-east-1 [docs]
```

`wwe-ptype` is in `us-east-1` (verified 2026-11-16: `get-bucket-location` returns `null`, which is how S3 reports us-east-1, and the bucket endpoint returns `x-amz-bucket-region: us-east-1`). Use `--region us-east-1` wherever a region is needed below.

### PBS batch jobs

Batch jobs on Derecho read the same home directory (shared GLADE filesystem), so they see the same
`~/.aws/credentials`, and `export AWS_PROFILE=ai2es-testbed-user` in the job script is enough. The catch is lifetime:
what determines it is the **session duration chosen when the temporary keys were issued** (on the
role / permission set that issued them), not anything on Derecho. A job that
starts its upload after that moment fails with `ExpiredToken`, and so does a job that is still uploading when it passes. I do not know that
duration for your account; the access portal or the account administrator can tell you. Practical consequences:

- Refresh the keys before submitting a run that will finish later than the validity window.
- Make the first command of the upload step `aws sts get-caller-identity` so a job fails immediately and clearly instead of after hours of inference.
- If this becomes routine (every forecast cycle), long-lived keys on a dedicated IAM user limited to this bucket, or a role the jobs can assume, are
  better than pasting temporary keys. That is an account administration decision.

## 2. Configure the bucket (run once)

Files: `docs/aws/wwe-ptype-bucket-policy.json`, `docs/aws/wwe-ptype-cors.json`.

### 2.1 Block Public Access

AWS treats a bucket policy that grants access to `Principal: "*"` as public, even with an `aws:Referer` condition,
**[docs]** and `put-bucket-policy` is rejected while *BlockPublicPolicy* is on; *RestrictPublicBuckets* would also
make the policy ineffective for anonymous requests. **[docs]** Only those two need to be off; keep the ACL protections on:

```bash
aws s3api get-public-access-block --bucket wwe-ptype      # see what is set now
aws s3api put-public-access-block --bucket wwe-ptype --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=false,RestrictPublicBuckets=false
```

Account-level Block Public Access settings override the bucket's. **[docs]** Check them (this may need permissions your user lacks, and
may be controlled by your organization's AWS administrator):

```bash
aws s3control get-public-access-block --account-id "$(aws sts get-caller-identity --query Account --output text)"
```

Your organization may forbid public buckets; if the commands are denied, that is the answer and it is not something to work around.
Also **[docs]**: anonymous reads do not work on objects encrypted with SSE-KMS (default SSE-S3 is fine).

### 2.2 Bucket policy

```bash
aws s3api put-bucket-policy --bucket wwe-ptype --policy file://docs/aws/wwe-ptype-bucket-policy.json
aws s3api get-bucket-policy --bucket wwe-ptype --query Policy --output text
```

The policy allows only `s3:GetObject` on `arn:aws:s3:::wwe-ptype/*`, and only when the `Referer` request header matches
`https://ai2es.github.io/*`, `http://localhost:*/*` or `http://127.0.0.1:*/*`. It does not grant `s3:ListBucket`, so the
bucket cannot be listed anonymously. Uploads and listing keep working for your IAM user through its own permissions.

Why the patterns have that shape: for a cross-origin `fetch`, the browser's default referrer policy
(`strict-origin-when-cross-origin`) sends only the **origin** with a trailing slash, i.e. `Referer: https://ai2es.github.io/`
(not the page path). **[docs]** (browser behaviour per the Referrer-Policy spec/MDN). `https://ai2es.github.io/*` matches that and
also longer forms. For local dev the origin includes the port, hence `http://localhost:*/*`. If the site (or a
`<meta name="referrer">`) ever sets `no-referrer` or `same-origin`, the header is not sent and every request will be
denied. The site code in this checkout sets no referrer policy.

A consequence of having no `ListBucket`: for a key that does not exist, S3 answers **403, not 404** to anonymous
requests. **[docs]** The site's zarr reader (`@zarrita/storage` `FetchStore`, whose source I read) returns "missing" only for 404 and
throws on 403, and its own documentation says to provide a custom `fetch` to remap this. Anything the reader probes that is absent
(for example a chunk of a never-written region, or an optional file) will therefore throw. Remapping 403 to 404 in the
site's `fetch` is the cleaner fix; granting `ListBucket` would reintroduce listing.

### 2.3 CORS

```bash
aws s3api put-cors-configuration --bucket wwe-ptype --cors-configuration file://docs/aws/wwe-ptype-cors.json
aws s3api get-cors-configuration --bucket wwe-ptype
```

Allowed origins: `https://ai2es.github.io` (no trailing slash, no path: origins are scheme + host + port) and
`http://localhost:*` / `http://127.0.0.1:*`. **[docs]** An `AllowedOrigins` entry may contain at most one `*` wildcard, so one pattern per host covers any dev port.
Methods `GET`, `HEAD`. `AllowedHeaders: ["*"]` lets any requested header through the preflight, which
includes `Range`; with only GET/HEAD and fixed origins this is not a meaningful widening. Exposed headers `ETag`,
`Content-Length`, `Content-Range`, `Accept-Ranges` are the ones a Range-reading client needs to see. `MaxAgeSeconds` is how long
the browser may cache a preflight answer.

### 2.4 What this does and does not protect

The `Referer` header is set by the client. A browser will not let a page forge it, but `curl -H "Referer: https://ai2es.github.io/"`
does, and anyone who sees the pattern (it is in this repo, and visible in browser dev tools) can do the same. This keeps the data off
search engines and away from casual or accidental use. **It is not access control.** Anyone determined can read all the data, and
you pay for the data-transfer. If the data must be restricted for real:

- **CloudFront with signed cookies or URLs**, bucket kept private behind origin access control. Real authentication, but needs
  something to issue the cookies (a small login endpoint).
- **A small authenticating proxy** (Lambda / API Gateway or a service on a server) that checks credentials and streams byte ranges.
- **Presigned URLs** minted by such a service; short-lived, but zarr reads many objects so they must be refreshed.

## 3. Upload a run and update the index

Do these in this order: the stores first, the index last, so the site never lists a run whose data is not there yet.

```bash
export AWS_PROFILE=ai2es-testbed-user
INIT=20240430_0000            # YYYYMMDD_HH00
Y=2024; M=04; D=30
LOCAL=/path/to/output         # holds hrrr_ptype_${INIT}.zarr and hrrr_soundings_${INIT}.zarr

aws sts get-caller-identity > /dev/null || exit 1
aws s3 sync "$LOCAL/hrrr_ptype_${INIT}.zarr"      "s3://wwe-ptype/ptype/$Y/$M/$D/hrrr_ptype_${INIT}.zarr"           --only-show-errors
aws s3 sync "$LOCAL/hrrr_soundings_${INIT}.zarr"  "s3://wwe-ptype/soundings/$Y/$M/$D/hrrr_soundings_${INIT}.zarr"   --only-show-errors
```

- `--only-show-errors` keeps a store with thousands of small chunk files from flooding the PBS log; errors still show and give a non-zero exit status.
- Content types: the CLI guesses from the file extension. **[docs]** `zarr.json` becomes `application/json`; chunk files have no extension and are stored
  as a generic binary type, which is fine for `fetch(...).arrayBuffer()`. Do not set `Content-Encoding` on chunks: Zarr already compresses them, and the browser would try to decode twice.
- **No `--delete`.** With it, `sync` removes every object under the destination that is not in the source. A mistyped
  destination prefix, or a partly written local store, would silently delete good data, and nothing here is versioned. A re-run overwrites
  changed files without it. If a run really needs cleaning, delete that one prefix explicitly with `aws s3 rm --recursive` after looking at it.
  (`sync` skips files whose size and timestamp say they are unchanged; **[docs]**.)

Then add the run to the index. The conditional update means several jobs can do this at once without losing each other's entries:

```bash
python -m ptype.hrrr_fast.index add wwe-ptype 2024-04-30T00:00 1-18 --region us-east-1
```

`--profile` defaults to `$AWS_PROFILE` and is read through `boto3`, which must be installed in the environment (obstore's own S3 client does not read AWS profiles).
From Python:

```python
from datetime import datetime
from obstore.store import S3Store
from ptype.hrrr_fast.index import INDEX_KEY, make_entry, update_index
update_index(S3Store("wwe-ptype", region="us-east-1"), INDEX_KEY, make_entry(datetime(2024, 4, 30, 0), list(range(1, 19))))
```

Index format (backward compatible with what the site reads: `version`, `inits[].init`, `forecast_hours`, `store`):

```json
{ "version": 1, "updated": "2024-04-30T01:12:09Z",
  "inits": [ { "init": "2024-04-30T00:00Z", "forecast_hours": [1, 2, 3],
               "store": "2024/04/30/hrrr_ptype_20240430_0000.zarr",
               "soundings": "soundings/2024/04/30/hrrr_soundings_20240430_0000.zarr" } ] }
```

`store` is relative to `ptype/`, as before. `soundings` is relative to the **bucket root** (or may be an absolute URL), because the soundings are
a sibling of `ptype/`. The site does not read `soundings` yet; adding it is a separate site change. Re-adding a run replaces its entry wholesale (hours are not merged).

How the concurrency safety works: each update reads `inits.json` with its ETag and writes back with an `If-Match` on that ETag (or `If-None-Match: *` if the file does not yet exist),
retrying with jittered backoff on failure. **Verified here:** obstore 0.12.1 exposes this as `put(..., mode="create")` and `put(..., mode={"e_tag": ...})`, raising
`AlreadyExistsError` / `PreconditionError`; `MemoryStore` enforces it, and the tests in `tests/test_hrrr_fast_index.py` run 40 threads against it with no lost entries.
`LocalStore` raises `NotImplementedError` for the update mode. **Not tested:** against real S3. **[docs]** S3 supports conditional writes with `If-None-Match` and `If-Match` on PutObject, and
obstore's S3 `conditional_put` setting defaults to `"etag"`, i.e. these headers. If S3 returned an error other than a precondition failure (for example a 409 for concurrent conflicting writes
**[docs]**), the function would raise instead of retrying; re-run the command in that case.

### Verify from a laptop

Use any object you know exists.

```bash
URL=https://wwe-ptype.s3.us-east-1.amazonaws.com/ptype/inits.json

# (a) no Referer: expect 403
curl -s -o /dev/null -w "%{http_code}\n" "$URL"

# (b) website Referer: expect 200
curl -s -o /dev/null -w "%{http_code}\n" -H "Referer: https://ai2es.github.io/" "$URL"

# (c) CORS preflight, then a Range request (expect 204/200 with access-control-allow-* headers, then 206 with content-range)
curl -si -X OPTIONS "$URL" -H "Origin: https://ai2es.github.io" -H "Access-Control-Request-Method: GET" -H "Access-Control-Request-Headers: range" | head -20
curl -si "$URL" -H "Origin: https://ai2es.github.io" -H "Referer: https://ai2es.github.io/" -H "Range: bytes=0-99" | head -20
```

`curl http://localhost` forms work the same with `Origin: http://localhost:5173` and `Referer: http://localhost:5173/`. Expected statuses are what AWS documents, not yet observed for this bucket.
Finally open the site (and `npm run dev`) and check the browser console and Network tab.

## 4. Troubleshooting

- **`ExpiredToken` / "security token included in the request is expired"**: the temporary keys in `~/.aws/credentials` ran out. Refresh them (section 1).
  Same message from a PBS job means they expired before or during the job.
- **`AccessDenied` on `aws s3 ls s3://wwe-ptype/`** (no prefix) or anonymous listing: expected for anonymous users, because the policy grants no `ListBucket`. For your
  own identity, it means your IAM permissions lack `s3:ListBucket`; try `aws s3 ls s3://wwe-ptype/ptype/` and ask the administrator.
  `AccessDenied` on `put-bucket-policy` with "BlockPublicPolicy" in the message: do step 2.1 (or your organization forbids it).
- **Browser console: "blocked by CORS policy"**: check `get-cors-configuration`; the origin must match exactly (scheme, host, port; no trailing slash). A CORS error is also what the browser shows
  for a 403 response, because S3's error response may lack CORS headers, so also test with curl (b) to separate a Referer/policy problem from a CORS problem.
- **403 in the Network tab for objects that exist**: the `Referer` did not match (check the request headers in dev tools; a `Referrer-Policy: no-referrer` setting would remove it), or the object is SSE-KMS encrypted.
- **403 for an object that does not exist**: expected (see 2.2); S3 hides absence from callers without `ListBucket`. **[docs]**
- **Index update raises `RuntimeError: could not update ...`**: many consecutive lost races; re-run. Check that `inits.json` is valid JSON if it fails to parse.
