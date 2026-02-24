#!/usr/bin/env python3
import json
from util import cancel_task

JOB_ID = 'sync-7bf0fb91-c4df-4ca8-b0ed-c91de1af29ca-e1'


if __name__ == '__main__':
    r = cancel_task(JOB_ID)

    print(r.status_code)
    print(json.dumps(r.json(), indent=4, default=str))
