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
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

prs = api('/pulls?state=open&per_page=20')
for pr in sorted(prs, key=lambda x: x['number']):
    print(f"PR #{pr['number']}: [{pr['state']}] {pr['title']}")
    print(f"  base: {pr['base']['ref']} ← head: {pr['head']['ref']}")
    print(f"  mergeable: {pr['mergeable']}")
    print()
