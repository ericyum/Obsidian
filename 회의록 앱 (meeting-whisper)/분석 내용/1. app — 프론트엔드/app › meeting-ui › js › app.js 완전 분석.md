# app.js 완전 분석

> 소스: `app/meeting-ui/js/app.js`  
> 역할: 프론트엔드 오케스트레이터. 상태, 이벤트, 업로드, polling, 목록/상세 렌더링을 묶는다.

## 1. 이 파일이 있는 이유

프론트엔드 오케스트레이터. 상태, 이벤트, 업로드, polling, 목록/상세 렌더링을 묶는다.

이 파일은 **1. app — 프론트엔드** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `app/meeting-ui/js/app.js` |
| 형식 | `javascript` |
| 전체 줄 수 | 2974 |
| 주석 줄 수 | 13 |
| 주요 심볼 수 | 142 |
| 환경 변수 참조 수 | 0 |

## 3. 의존성

- `/js/recorder.js?v=20260630-ux-fixes`
- `/js/api.js?v=20260708-session`
- `/js/ui.js?v=20260710-uiux`
- `/js/drafts.js?v=20260630-ux-fixes`
- `/js/session.js?v=20260708-session`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `routeFromLocation` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `routeUrl` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `writeRoute` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `syncMeetingRoute` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `applyRoute` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `initUserProfile` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `userInitials` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `loadList` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `buildMeetingSearchText` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderMeetingListView` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `updateListCount` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderMeetingCategoryFilter` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `knownCategories` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `restoreListFiltersFromUrl` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `syncListFiltersToUrl` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `resetListFilters` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `bindListEvents` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `openDialog` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `showErrorDialog` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `isTypingIn` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `paintDraftBanner` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `downloadRecoveredDraft` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `uploadRecoveredDraft` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderEntryCategoryOptions` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `navigateBackToList` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `selectedMeetingCategory` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `seoulDateStamp` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `nextCategoryTitle` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `updateTitleFromCategory` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `syncCategoryInput` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderChips` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `moveArrayItem` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderEntryPeopleChips` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `chipDragItems` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `moveChipFromDragData` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `bindDraggableChips` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `bindChipDropZone` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `isChipDragEvent` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `draggedChipData` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `addSuggestionValue` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `knownPeopleValues` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `updatePeopleSuggestions` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `attachPeopleMultiSelect` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `parseSpeakerBound` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `isDiarizeEnabled` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `syncDiarizeFields` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `transcriptionOptionsFromForm` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `speakerBoundsFromForm` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `showEntryError` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `micPermissionMessage` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `audioExtFromFile` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `titleFromFile` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `formatBytes` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `readMediaDuration` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `selectedAudioFile` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `setStartButtonLabel` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `syncFileUploadCta` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `setUploadInFlight` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderShareChips` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `uploadSelectedFile` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `promptRecordingBackup` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `makePendingUpload` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `retryPendingUpload` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `updateRecordingTimer` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `setRecordingStatus` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `initMicGraph` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `updateMicLevel` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `startLiveTranscript` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `stopLiveTranscript` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `liveFinalLines` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `saveLiveFinalToDraft` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `syncLiveFinalFromEditor` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `bindLiveFinalEditors` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `appendLiveLine` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `commitLiveInterim` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `paintLiveTranscript` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `resetResultView` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `paintResult` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `paintInlineStatus` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderAccessPanel` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderAccessCategoryEditor` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderAccessChips` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `bindAccessPanelEvents` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `saveMeetingCategory` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `saveMeetingAccess` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `applyReadOnlyMode` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `openMeeting` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `poll` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `stopPolling` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `showDownloads` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `openSummaryInOutlook` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `transcriptToText` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `downloadTranscriptText` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `copySummaryMarkdown` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `summaryMarkdownDownload` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `downloadSummaryMarkdown` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `saveSummaryMarkdownAs` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `saveBlobToChosenLocation` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `chooseSaveTarget` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `writeBlobToSaveTarget` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `downloadBlobNormally` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `copyText` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `noteToMarkdown` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `safeFilename` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `openSpeakerModal` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `closeSpeakerModal` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderSegmentSpeakerChoices` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `assignSegmentSpeaker` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `knownSpeakers` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `compareSpeakerIds` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `speakerIndex` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `noteDelBtn` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderSummaryPlaceholder` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderInlineSummary` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderInlineTranscript` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderSummaryReview` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderReviewBusyAction` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `bindInlineNoteEditor` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `generateReviewCandidates` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `removeNoteItem` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `bindInlineTranscriptEditor` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `applyReviewItem` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `dismissReviewItem` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `dismissAllReviewItems` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `applyAllReviewSuggestions` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `applyReviewReplacementToNote` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `saveReplacedNote` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `scheduleNoteSave` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `addNoteItem` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `saveNoteInline` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `scheduleTranscriptSave` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `saveTranscriptInline` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `rerunFromInline` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `collectNoteInline` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `collectTranscriptInline` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `inlineText` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `setInlineMessage` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderInlineStatusText` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `renderRerunAction` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `retryCurrentMeeting` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `keepPlainEditable` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `showResErr` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 6. 경로·엔드포인트 단서

- `/js/recorder.js?v=20260630-ux-fixes`
- `/js/api.js?v=20260708-session`
- `/js/ui.js?v=20260710-uiux`
- `/js/drafts.js?v=20260630-ux-fixes`
- `/js/session.js?v=20260708-session`
- `/logout`

## 7. 코드 흐름 상세

### 7.1 `routeFromLocation`

- 위치: 73~78행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `routeUrl`

- 위치: 79~87행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `writeRoute`

- 위치: 88~96행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `syncMeetingRoute`

- 위치: 97~101행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `applyRoute`

- 위치: 102~116행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `initUserProfile`

- 위치: 117~156행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.7 `userInitials`

- 위치: 157~164행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.8 `loadList`

- 위치: 165~188행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.9 `buildMeetingSearchText`

- 위치: 189~210행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.10 `renderMeetingListView`

- 위치: 211~238행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.11 `updateListCount`

- 위치: 239~251행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.12 `renderMeetingCategoryFilter`

- 위치: 252~275행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.13 `knownCategories`

- 위치: 276~285행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.14 `restoreListFiltersFromUrl`

- 위치: 286~298행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.15 `syncListFiltersToUrl`

- 위치: 299~310행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.16 `resetListFilters`

- 위치: 311~321행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.17 `bindListEvents`

- 위치: 322~401행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.18 `openDialog`

- 위치: 402~452행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.19 `showErrorDialog`

- 위치: 453~461행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.20 `isTypingIn`

- 위치: 462~475행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.21 `paintDraftBanner`

- 위치: 476~521행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.22 `downloadRecoveredDraft`

- 위치: 522~554행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.23 `uploadRecoveredDraft`

- 위치: 555~616행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.24 `renderEntryCategoryOptions`

- 위치: 617~652행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.25 `navigateBackToList`

- 위치: 653~662행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.26 `selectedMeetingCategory`

- 위치: 663~668행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.27 `seoulDateStamp`

- 위치: 669~679행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.28 `nextCategoryTitle`

- 위치: 680~695행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.29 `updateTitleFromCategory`

- 위치: 696~706행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.30 `syncCategoryInput`

- 위치: 707~730행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.31 `renderChips`

- 위치: 731~751행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.32 `moveArrayItem`

- 위치: 752~761행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.33 `renderEntryPeopleChips`

- 위치: 762~767행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.34 `chipDragItems`

- 위치: 768~773행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.35 `moveChipFromDragData`

- 위치: 774~794행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.36 `bindDraggableChips`

- 위치: 795~837행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.37 `bindChipDropZone`

- 위치: 838~868행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.38 `isChipDragEvent`

- 위치: 869~874행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.39 `draggedChipData`

- 위치: 875~886행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.40 `addSuggestionValue`

- 위치: 887~892행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

## 8. 변경 시 영향 범위
+
- 공개 계약, 상태명, 환경 변수, URL을 바꾸면 호출 측과 테스트를 함께 수정해야 한다.
- 저장 데이터의 형식이 바뀌면 기존 HANA 행과 진행 중인 작업의 호환성을 확인한다.
- 인증·소유권 검사는 편의상 우회하면 안 된다. Approuter 인증과 CAP 역할 검사는 서로 다른 계층이다.

## 9. 관련 문서

- [[00. 전체 구조와 책임 지도]]
- [[01. 녹음부터 회의록까지 End-to-End]]
- [[02. 데이터 모델과 상태 전이]]
- [[03. 인증·권한·신뢰 경계]]
