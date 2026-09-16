# default-env.json 민감정보 커밋 해결 총정리

## 📌 문제 발생

OTT 프로젝트 진행 중 `default-env.json` 파일이 Git 커밋에 포함된 사실을 발견.
해당 파일에는 **라우팅 정보 + 민감한 서비스 바인딩 키값**이 포함되어 있어 보안상 즉시 조치가 필요했음.

### 문제가 된 파일 목록

| 파일 경로 | 내용 |
|---|---|
| `cap-node/core/default-env.json` | 백엔드 코어 설정 (서비스 키, destination 등) |
| `cap-node/ott/default-env.json` | 백엔드 OTT 설정 |
| `cap-app/approuter/dev/default-env.json` | 앱라우터 개발 환경 설정 |
| `cap-app/approuter_ott/dev/default-env.json` | OTT 앱라우터 개발 환경 설정 |

---

## 🔧 해결 과정 (2단계)

### 1단계: 각 팀원별 `git rm --cached` + `.gitignore` 등록

각 팀원이 자신의 feature 브랜치에서 아래 명령어 실행:

```bash
# 1. .gitignore에 default-env.json 등록 확인
grep "default-env" .gitignore

# 2. git 추적에서만 제거 (파일은 로컬에 그대로 유지)
git rm --cached cap-app/approuter_ott/dev/default-env.json
git rm --cached cap-app/approuter/dev/default-env.json
git rm --cached cap-node/core/default-env.json

# 3. 커밋 후 push
git commit -m "chore: untrack default-env.json"
git push
```

그 후 각자 **dev 브랜치로 PR 생성 → 머지**.

| 팀원 | 브랜치 | untrack 커밋 | PR | dev 머지 |
|---|---|---|---|---|
| juyum | feature/juyum | `08b8ba6` | #14 | ✅ |
| yina | feature/yina | `34e6bba` | #15 | ✅ |
| hsan | feature/hsan | `d3fd90b` (`.gitignore` 포함) | #16 | ✅ |

> `.gitignore` 마지막 줄에 `**/default-env.json` 추가됨 → 앞으로 트래킹 방지

### 1단계만으로는 부족했던 이유

`git rm --cached`는 **앞으로의 트래킹만 막을 뿐**, 과거 커밋 히스토리에 이미 기록된 파일 내용은 그대로 남아있음.
누구든 `git show <옛커밋해시>:cap-node/core/default-env.json` 명령어로 키값 조회 가능.

---

### 2단계: `git filter-branch`로 전체 히스토리 정리

1단계 머지 완료 후, **dev 브랜치 기준으로 `git filter-branch` 실행**하여 모든 과거 커밋에서 `default-env.json` 흔적을 완전 제거.

```bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch \
    cap-app/approuter_ott/dev/default-env.json \
    cap-app/approuter/dev/default-env.json \
    cap-node/core/default-env.json \
    cap-node/ott/default-env.json" \
  --prune-empty --tag-name-filter cat -- --all
```

> `--index-filter`: 각 커밋마다 인덱스에서 해당 파일 제거
> `--ignore-unmatch`: 파일이 없는 커밋은 무시
> `--prune-empty`: 빈 커밋이 된 경우 제거
> `--all`: 모든 브랜치, 태그, 원격 추적 브랜치 포함

처리된 커밋: **233개**

---

## 📤 Force Push

히스토리가 재작성되었으므로 모든 브랜치를 force push:

```bash
# 로컬 브랜치
git push --force origin dev feature/juyum

# 원격 전용 브랜치
git push --force origin \
  +refs/remotes/origin/feature/yina:feature/yina \
  +refs/remotes/origin/feature/hsan:feature/hsan \
  +refs/remotes/origin/master:master
```

마지막으로 오래된 객체 정리:

```bash
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

## ✅ 최종 검증 결과

**모든 브랜치에서 `default-env.json` 커밋 기록 제로.**

| 브랜치 | default-env.json 기록 |
|---|---|
| `origin/dev` | 없음 ✅ |
| `dev` (로컬) | 없음 ✅ |
| `origin/feature/juyum` | 없음 ✅ |
| `feature/juyum` (로컬) | 없음 ✅ |
| `origin/feature/yina` | 없음 ✅ |
| `origin/feature/hsan` | 없음 ✅ |
| `origin/master` | 없음 ✅ |

---

## 👥 팀원별 후속 조치

`filter-branch`로 **모든 커밋 해시가 변경**되었기 때문에, 각 팀원은 원격 히스토리에 로컬을 맞춰야 함.

### juyum
```bash
git fetch origin
git reset --hard origin/feature/juyum
```

### yina
```bash
git checkout feature/yina
git fetch origin
git reset --hard origin/feature/yina
```

### hsan
```bash
git checkout feature/hsan
git fetch origin
git reset --hard origin/feature/hsan
```

> ⚠️ 작업 중인 변경사항이 있으면 `git stash` → `reset` → `git stash pop` 순서로

### 다시 하지 않아도 되는 것
- ❌ `git rm --cached default-env.json` → 이미 히스토리에서 완전 제거됨
- ❌ `git commit -m "chore: untrack..."` → `.gitignore`에 이미 등록됨
- ❌ 머지 → 각 브랜치가 이미 force push되어 최신 상태

---

## ⏱ 타임라인

| 시간 | 작업 |
|---|---|
| ~6/8 | 초기 커밋에 `default-env.json` 포함됨 (`50be27c`, `d084315`) |
| 6/8~6/24 | 여러 커밋에서 파일 수정/추가 (`68b0453`, `19f4563`, `e133b02`, `f7d5cd8`, `1970fe4`, `2ecf13c` 등) |
| 6/24 | juyum이 `.gitignore`에 등록 + `rm --cached` (`08b8ba6` → PR #14) |
| 6/25 오전 8:35 | yina `rm --cached` (`34e6bba` → PR #15) |
| 6/25 오전 9:04 | hsan `.gitignore` + `rm --cached` (`d3fd90b` → PR #16) |
| 6/25 오전 | PR #14, #15, #16 dev에 머지 완료 |
| 6/25 오후 | `git filter-branch` 실행 → 전체 히스토리 정리 → force push |

---

## 🔐 보안 권고

1. `default-env.json`은 `.gitignore`에 영구 등록 (`**/default-env.json`)
2. 실제 키값은 환경변수 또는 BTP 서비스 바인딩으로 관리
3. `.env` 파일도 `.gitignore`에 포함되어 있음
