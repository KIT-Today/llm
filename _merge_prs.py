 import urllib.request, json

token = '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'

def api(path, method='GET', data=None):
    req = urllib.request.Request(
        f'https://api.github.com/repos/KIT-Today/llm{path}',
        data=json.dumps(data).encode() if data else None,
        headers={
            'Authorization': 'token ' + token,
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        },
        method=method
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        return None, f"{e.code}: {e.read().decode()}"

# 오픈 PR 목록
prs, err = api('/pulls?state=open&per_page=20')
print("=== Open PRs ===")
for pr in sorted(prs, key=lambda x: x['number']):
    print(f"  PR #{pr['number']}: {pr['title'][:50]}")
    print(f"    base: {pr['base']['ref']} ← {pr['head']['ref']}")

# 머지 순서: #10 먼저 (main 기반), #9 나중 (충돌 가능성 있음)
merge_order = [10, 9]

print("\n=== Merging ===")
for pr_num in merge_order:
    # 개별 PR 상세 조회 (mergeable 확인)
    pr, err = api(f'/pulls/{pr_num}')
    if err:
        print(f"PR #{pr_num}: 조회 실패 - {err}")
        continue

    title = pr['title'][:50]
    mergeable = pr.get('mergeable')
    print(f"\nPR #{pr_num}: {title}")
    print(f"  mergeable: {mergeable}")

    if mergeable is False:
        print(f"  ❌ 충돌 있음 - 스킵")
        continue

    result, err = api(f'/pulls/{pr_num}/merge', method='PUT', data={
        'merge_method': 'squash',
        'commit_title': pr['title'],
        'commit_message': f"Squashed from PR #{pr_num}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    })

    if err:
        print(f"  ❌ 머지 실패: {err}")
    else:
        print(f"  ✅ 머지 완료: {result.get('sha', '')[:8]}")
