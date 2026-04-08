# -*- coding: utf-8 -*-
"""
상수 및 매핑 테이블
"""

from prompts import PersonaType

# ============================================
# 페르소나 매핑
# ============================================

PERSONA_MAP = {
    # 문자열 키
    "warm_counselor": PersonaType.WARM_COUNSELOR,
    "practical_advisor": PersonaType.PRACTICAL_ADVISOR,
    "friendly_buddy": PersonaType.FRIENDLY_BUDDY,
    "calm_mentor": PersonaType.CALM_MENTOR,
    "cheerful_supporter": PersonaType.CHEERFUL_SUPPORTER,
    # 숫자 키 (백엔드 호환)
    1: PersonaType.WARM_COUNSELOR,
    2: PersonaType.PRACTICAL_ADVISOR,
    3: PersonaType.FRIENDLY_BUDDY,
    4: PersonaType.CALM_MENTOR,
    5: PersonaType.CHEERFUL_SUPPORTER,
}


# ============================================
# 활동 카테고리 매핑
# ============================================

BURNOUT_TO_ACTIVITY_CATEGORY = {
    "정서적_고갈": ["REST", "SMALL_WIN"],
    "좌절_압박": ["VENTILATION", "REST"],
    "부정적_대인관계": ["VENTILATION", "SMALL_WIN"],
    "자기비하": ["SMALL_WIN", "REST"],
}

ACTIVITY_CATEGORY_IDS = {
    "REST": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    "VENTILATION": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75],
    "SMALL_WIN": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
}

ACTIVITY_CONTENT = {
    1: "따뜻한 차/코코아 한 잔 마시기",
    2: "좋아하는 향초/인센스 피우기",
    3: "5분 동안 눈 감고 호흡에 집중하기",
    4: "스마트폰 끄고 1시간 동안 디지털 디톡스",
    5: "따뜻한 물로 샤워하거나 반신욕 하기",
    6: "좋아하는 ASMR 듣기 (빗소리, 장작 등)",
    7: "수면 안대 쓰고 20분 낮잠 자기",
    8: "창문 열고 신선한 공기 마시기",
    9: "공원 벤치에 앉아 햇볕 쬐기",
    10: "조용한 카페에서 멍때리기",
    11: "천천히 동네 한 바퀴 산책하기",
    12: "숲이나 나무가 많은 곳 걷기(삼림욕)",
    13: "잔잔한 클래식/재즈 음악 감상",
    14: "좋아하는 동물 보기(영상/실물)",
    15: "찜질방/사우나 가서 땀 빼기",
    16: "코인 노래방 가서 소리 지르기",
    17: "이면지에 낙서하고 박박 찢어버리기",
    18: "매운 음식 먹고 땀 흘리기",
    19: "베개에 얼굴 묻고 소리치기",
    20: "빠른 비트의 댄스 음악 듣기",
    21: "오락실 펀치 기계/두더지 잡기",
    22: "숨이 찰 때까지 3분만 전력 질주",
    23: "공포 영화나 스릴러 영화 보기",
    24: "방 청소/정리하며 몸 움직이기",
    25: "친구에게 전화해 하소연하기",
    26: "사람 많은 번화가 구경하기",
    27: "배팅장에서 야구 공 치기",
    28: "유튜브에서 '웃긴 영상' 모음 보기",
    29: "아이스 음료 벌컥벌컥 마시기",
    30: "PC방/집에서 게임 한 판 하기",
    31: "일어나자마자 이불 개기",
    32: "물 한 잔 시원하게 마시기",
    33: "책상 위 지저분한 것 3개만 치우기",
    34: "스마트폰 갤러리/스크린샷 정리하기",
    35: "읽지 않은 스팸 메일/문자 삭제하기",
    36: "영양제/비타민 챙겨 먹기",
    37: "거울 닦기 / 화장실 세면대 닦기",
    38: "5분 스트레칭하기",
    39: "하늘 사진 예쁘게 1장 찍기",
    40: "편의점에서 좋아하는 간식 사 오기",
    41: "엘리베이터 대신 계단 이용하기",
    42: "다 쓴 물건 제자리에 놓기",
    43: "오늘 할 일(To-do) 1개만 적어보기",
    44: "길가에 떨어진 쓰레기 1개 줍기",
    45: "식물에 물 주기",

    # ── REST 추가 (46~60) ──────────────────────────────────────
    46: "좋아하는 음악 플레이리스트 틀어놓고 멍때리기",
    47: "따뜻한 핫팩 손에 쥐고 잠깐 눈 감기",
    48: "유튜브 자연 영상(숲, 파도) 틀어놓기",
    49: "좋아하는 책 한 페이지만 읽기",
    50: "침대에 누워 천장 바라보기 (5분)",
    51: "좋아하는 향수/바디로션 바르기",
    52: "따뜻한 물로 손·발 담그기",
    53: "창밖 풍경 5분 바라보기",
    54: "스트레칭 영상 따라 하기 (10분)",
    55: "넷플릭스/유튜브에서 좋아하는 예능 한 편 보기",
    56: "조용히 일기 쓰기 (오늘 감사한 것 3가지)",
    57: "마음챙김 명상 앱 5분 실행하기",
    58: "따뜻한 수프/라면 한 그릇 먹기",
    59: "좋아하는 웹툰/만화 한 화 읽기",
    60: "하늘 보며 구름 모양 찾기",

    # ── VENTILATION 추가 (61~75) ────────────────────────────────
    61: "쿠션이나 이불 힘껏 치기",
    62: "드라이브하며 큰 소리로 노래 부르기",
    63: "종이에 화나는 것 다 쓰고 찢기",
    64: "빠른 템포 음악 틀고 10분 맘대로 움직이기",
    65: "냉동 과일 씹어 먹으며 스트레스 해소",
    66: "수영장 가서 물속에서 소리 지르기",
    67: "계단 오르내리기 3번 반복",
    68: "좌절/압박 감정을 그림으로 그려보기",
    69: "샌드백 or 쿠션에 킥복싱 흉내 내기",
    70: "신나는 게임 BGM 틀고 집안일 하기",
    71: "스쿼트 10개 or 팔굽혀펴기 5개 하기",
    72: "화나는 감정 보이스메모로 혼자 녹음하기",
    73: "마트 or 편의점 목적 없이 돌아다니기",
    74: "유튜브 먹방 영상 보기",
    75: "볼풀공/스트레스볼 힘껏 쥐었다 펴기",

    # ── SMALL_WIN 추가 (76~90) ──────────────────────────────────
    76: "책상 서랍 하나만 정리하기",
    77: "스마트폰 배경화면 바꾸기",
    78: "오늘 마신 물 양 체크하기",
    79: "안 쓰는 앱 3개 삭제하기",
    80: "내일 입을 옷 미리 골라두기",
    81: "좋아하는 유튜버 영상에 댓글 하나 달기",
    82: "지갑 영수증 정리하기",
    83: "이불 커버 교체하기",
    84: "냉장고 유통기한 지난 것 버리기",
    85: "오늘 잘한 일 하나 소리 내어 말하기",
    86: "짧은 감사 문자 가족/친구에게 보내기",
    87: "내가 좋아하는 것 목록 5개 적어보기",
    88: "신발장 정리하기",
    89: "책상 위 먼지 닦기",
    90: "오늘 하루 일정 간단히 메모하기",
}


# ============================================
# 활동 속성 매핑 (백엔드 콜백용)
# act_category: REST / VENTILATION / SMALL_WIN
# is_active  : 신체 활동 여부
# is_outdoor : 야외 활동 여부
# is_social  : 사회적 교류 여부
# ============================================

ACTIVITY_ATTRIBUTES = {
    # ── REST (1~15) ──────────────────────────────────────
    1:  {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    2:  {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    3:  {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    4:  {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    5:  {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    6:  {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    7:  {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    8:  {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    9:  {"act_category": "REST", "is_active": False, "is_outdoor": True,  "is_social": False},
    10: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    11: {"act_category": "REST", "is_active": True,  "is_outdoor": True,  "is_social": False},
    12: {"act_category": "REST", "is_active": True,  "is_outdoor": True,  "is_social": False},
    13: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    14: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    15: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    # ── VENTILATION (16~30) ──────────────────────────────
    16: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    17: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    18: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    19: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    20: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    21: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    22: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": True,  "is_social": False},
    23: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    24: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    25: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": True},
    26: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": True,  "is_social": False},
    27: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    28: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    29: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    30: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    # ── SMALL_WIN (31~45) ────────────────────────────────
    31: {"act_category": "SMALL_WIN", "is_active": True,  "is_outdoor": False, "is_social": False},
    32: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    33: {"act_category": "SMALL_WIN", "is_active": True,  "is_outdoor": False, "is_social": False},
    34: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    35: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    36: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    37: {"act_category": "SMALL_WIN", "is_active": True,  "is_outdoor": False, "is_social": False},
    38: {"act_category": "SMALL_WIN", "is_active": True,  "is_outdoor": False, "is_social": False},
    39: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": True,  "is_social": False},
    40: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": True,  "is_social": False},
    41: {"act_category": "SMALL_WIN", "is_active": True,  "is_outdoor": False, "is_social": False},
    42: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    43: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    44: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": True,  "is_social": False},
    45: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},

    # ── REST 추가 (46~60) ──────────────────────────────────────
    46: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    47: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    48: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    49: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    50: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    51: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    52: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    53: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    54: {"act_category": "REST", "is_active": True,  "is_outdoor": False, "is_social": False},
    55: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    56: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    57: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    58: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    59: {"act_category": "REST", "is_active": False, "is_outdoor": False, "is_social": False},
    60: {"act_category": "REST", "is_active": False, "is_outdoor": True,  "is_social": False},

    # ── VENTILATION 추가 (61~75) ────────────────────────────────
    61: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    62: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": True,  "is_social": False},
    63: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    64: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    65: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    66: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": True,  "is_social": False},
    67: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    68: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    69: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    70: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    71: {"act_category": "VENTILATION", "is_active": True,  "is_outdoor": False, "is_social": False},
    72: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    73: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": True,  "is_social": False},
    74: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},
    75: {"act_category": "VENTILATION", "is_active": False, "is_outdoor": False, "is_social": False},

    # ── SMALL_WIN 추가 (76~90) ──────────────────────────────────
    76: {"act_category": "SMALL_WIN", "is_active": True,  "is_outdoor": False, "is_social": False},
    77: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    78: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    79: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    80: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    81: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": True},
    82: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    83: {"act_category": "SMALL_WIN", "is_active": True,  "is_outdoor": False, "is_social": False},
    84: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    85: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    86: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": True},
    87: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
    88: {"act_category": "SMALL_WIN", "is_active": True,  "is_outdoor": False, "is_social": False},
    89: {"act_category": "SMALL_WIN", "is_active": True,  "is_outdoor": False, "is_social": False},
    90: {"act_category": "SMALL_WIN", "is_active": False, "is_outdoor": False, "is_social": False},
}


# ============================================
# 감정 키워드 매핑
# ============================================

ENERGY_TO_EMOTION_MAP = {
    "답답하고 화나요": ["좌절_압박", "부정적_대인관계"],
    "지치고 무기력해요": ["정서적_고갈"],
    "불안하고 걱정돼요": ["자기비하", "좌절_압박"],
    "괜찮거나 기뻐요": ["긍정"]
}

# 백엔드 keywords 딕셔너리의 "나의 유형" 필드값 → 내부 카테고리 매핑
USER_TYPE_TO_EMOTION = {
    "긍정": "긍정",
    "정서적 고갈": "정서적_고갈",
    "좌절/압박": "좌절_압박",
    "부정적 대인관계": "부정적_대인관계",
    "자기 비하": "자기비하",
    "자기비하": "자기비하",
}

# 백엔드 keywords 딕셔너리의 "감정" 필드값 → 내부 카테고리 매핑
FEELING_TO_EMOTION = {
    "기쁨": "긍정", "즐거움": "긍정", "행복함": "긍정",
    "지침": "정서적_고갈", "방전됨": "정서적_고갈", "무기력": "정서적_고갈",
    "아무것도 안 함": "정서적_고갈", "의미 없음": "정서적_고갈", "무감각": "정서적_고갈",
    "답답함": "좌절_압박", "짜증남": "좌절_압박", "화남": "좌절_압박",
    "다 싫음": "부정적_대인관계", "혼자 있고 싶음": "부정적_대인관계",
    "불안함": "자기비하", "울고 싶음": "자기비하", "내가 싫음": "자기비하",
}

# 사용자 성향 추론 키워드 (user_text + keywords에서 매칭)
# prefer: 해당 속성 선호 신호 / avoid: 비선호 신호
PREFERENCE_SIGNAL_KEYWORDS = {
    "is_active": {
        "prefer": ["산책", "운동", "뛰", "움직", "몸 쓰"],
        "avoid": ["눕고", "쉬고", "아무것도 하기 싫"],
    },
    "is_outdoor": {
        "prefer": ["밖", "나가고", "공원", "바깥", "환기"],
        "avoid": ["나가기 싫", "집에", "방에", "밖은"],
    },
    "is_social": {
        "prefer": ["친구", "누군가", "같이", "사람이 보고"],
        "avoid": ["혼자", "사람 싫", "아무도", "연락 끊"],
    },
}

DETAIL_KEYWORD_TO_EMOTION = {
    # 답답하고 화나요
    "억울": "좌절_압박", "불만": "좌절_압박", "원망": "부정적_대인관계",
    "분노": "좌절_압박", "불평": "좌절_압박", "불편함": "좌절_압박",
    "불쾌": "부정적_대인관계", "질투": "부정적_대인관계", "당황": "좌절_압박",
    # 지치고 무기력해요
    "스트레스": "정서적_고갈", "무력함": "정서적_고갈", "공허함": "정서적_고갈",
    "답답함": "정서적_고갈", "부담": "좌절_압박", "지루함": "정서적_고갈",
    "귀찮음": "정서적_고갈", "후회": "자기비하", "외로움": "부정적_대인관계",
    # 불안하고 걱정돼요
    "불안": "자기비하", "걱정": "자기비하", "초조": "좌절_압박",
    "자괴감": "자기비하", "죄책감": "자기비하", "혼란함": "정서적_고갈",
    "조마조마함": "좌절_압박", "두려움": "자기비하", "부끄러움": "자기비하",
    # 괜찮거나 기뻐요
    "홀가분함": "긍정", "고마움": "긍정", "뿌듯함": "긍정",
    "의욕": "긍정", "설렘": "긍정", "기대": "긍정",
    "기쁨": "긍정", "즐거움": "긍정", "희망": "긍정"
}

SITUATION_KEYWORDS = {
    "사람 관계가 힘들어요": ["상명하복", "사내 정치", "동료 갈등"],
    "업무 방식이 문제예요": ["과도한 업무량", "잦은 야근", "불합리한 일처리"],
    "보상과 미래가 안 보여요": ["낮은 보상", "성장 정체", "의미 없는 반복"],
    "외부 요인으로 지쳐요": ["상사/고객의 갑질", "출퇴근 전쟁", "폭언"]
}


# ============================================
# 카테고리 정의
# ============================================

STAGE1_CATEGORIES = {0: "긍정", 1: "부정"}
STAGE2_CATEGORIES = {0: "정서적_고갈", 1: "좌절_압박", 2: "부정적_대인관계", 3: "자기비하"}

MBI_CATEGORY_MAP = {
    "긍정": "NONE",
    "정서적_고갈": "EMOTIONAL_EXHAUSTION",
    "좌절_압박": "FRUSTRATION_PRESSURE",
    "부정적_대인관계": "NEGATIVE_RELATIONSHIP",
    "자기비하": "SELF_DEPRECATION"
}

BURNOUT_KEYWORDS = {
    "긍정": {"keywords": ["좋다", "좋아", "행복", "기쁘", "뿌듯", "만족", "감사", "고맙", "다행", "홀가분"]},
    "정서적_고갈": {"keywords": ["지치", "피곤", "힘들", "무기력", "탈진", "녹초", "방전", "우울", "슬프", "귀찮"]},
    "좌절_압박": {"keywords": ["화나", "짜증", "열받", "분노", "억울", "압박", "스트레스", "답답", "한계"]},
    "부정적_대인관계": {"keywords": ["무시", "소외", "배신", "갈등", "서운", "외로", "실망", "오해"]},
    "자기비하": {"keywords": ["못하", "부족", "무능", "한심", "불안", "자책", "후회", "실패"]},
}
