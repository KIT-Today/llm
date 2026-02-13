# -*- coding: utf-8 -*-
"""
🚨 에러 코드 정의
=================

AI 서버 에러 코드 체계
- AI1xxx: 모델 관련 에러
- AI2xxx: 입력 데이터 관련 에러
- AI3xxx: 분석 처리 관련 에러
- AI4xxx: 외부 통신 관련 에러
- AI5xxx: 시스템 관련 에러
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ErrorCode(str, Enum):
    """에러 코드 정의"""
    # AI1xxx: 모델 관련
    MODEL_NOT_LOADED = "AI1001"
    STAGE1_INFERENCE_FAILED = "AI1002"
    STAGE2_INFERENCE_FAILED = "AI1003"
    EMBEDDING_FAILED = "AI1004"
    
    # AI2xxx: 입력 데이터 관련
    INVALID_PERSONA = "AI2001"
    EMPTY_HISTORY = "AI2002"
    INVALID_DIARY_FORMAT = "AI2003"
    CONTENT_TOO_SHORT = "AI2004"
    INVALID_KEYWORDS = "AI2005"
    
    # AI3xxx: 분석 처리 관련
    ANALYSIS_FAILED = "AI3001"
    FEEDBACK_GENERATION_FAILED = "AI3002"
    RECOMMENDATION_FAILED = "AI3003"
    EMOTION_MATCH_FAILED = "AI3004"
    INSIGHT_GENERATION_FAILED = "AI3005"
    
    # AI4xxx: 외부 통신 관련
    CALLBACK_FAILED = "AI4001"
    CALLBACK_TIMEOUT = "AI4002"
    BACKEND_UNREACHABLE = "AI4003"
    
    # AI5xxx: 시스템 관련
    INTERNAL_ERROR = "AI5001"
    OUT_OF_MEMORY = "AI5002"
    RATE_LIMITED = "AI5003"


class ErrorDetail(BaseModel):
    """에러 상세 정보"""
    code: str
    message: str
    detail: Optional[str] = None
    recoverable: bool = True
    fallback_used: bool = False


# 에러 정의 테이블
ERROR_DEFINITIONS = {
    ErrorCode.MODEL_NOT_LOADED: {
        "message": "AI 모델이 로드되지 않았습니다",
        "recoverable": False
    },
    ErrorCode.STAGE1_INFERENCE_FAILED: {
        "message": "1단계 감정 분류 실패",
        "recoverable": True
    },
    ErrorCode.STAGE2_INFERENCE_FAILED: {
        "message": "2단계 번아웃 분류 실패",
        "recoverable": True
    },
    ErrorCode.EMBEDDING_FAILED: {
        "message": "텍스트 임베딩 생성 실패",
        "recoverable": True
    },
    ErrorCode.INVALID_PERSONA: {
        "message": "유효하지 않은 페르소나",
        "recoverable": True
    },
    ErrorCode.EMPTY_HISTORY: {
        "message": "일기 히스토리가 비어있습니다",
        "recoverable": False
    },
    ErrorCode.INVALID_DIARY_FORMAT: {
        "message": "일기 형식이 올바르지 않습니다",
        "recoverable": True
    },
    ErrorCode.CONTENT_TOO_SHORT: {
        "message": "일기 내용이 너무 짧습니다",
        "recoverable": True
    },
    ErrorCode.INVALID_KEYWORDS: {
        "message": "키워드 형식이 올바르지 않습니다",
        "recoverable": True
    },
    ErrorCode.ANALYSIS_FAILED: {
        "message": "감정 분석 실패",
        "recoverable": True
    },
    ErrorCode.FEEDBACK_GENERATION_FAILED: {
        "message": "피드백 생성 실패",
        "recoverable": True
    },
    ErrorCode.RECOMMENDATION_FAILED: {
        "message": "활동 추천 생성 실패",
        "recoverable": True
    },
    ErrorCode.EMOTION_MATCH_FAILED: {
        "message": "감정 일치도 검사 실패",
        "recoverable": True
    },
    ErrorCode.INSIGHT_GENERATION_FAILED: {
        "message": "통계 인사이트 생성 실패",
        "recoverable": True
    },
    ErrorCode.CALLBACK_FAILED: {
        "message": "백엔드 콜백 전송 실패",
        "recoverable": False
    },
    ErrorCode.CALLBACK_TIMEOUT: {
        "message": "백엔드 콜백 타임아웃",
        "recoverable": False
    },
    ErrorCode.BACKEND_UNREACHABLE: {
        "message": "백엔드 서버 연결 불가",
        "recoverable": False
    },
    ErrorCode.INTERNAL_ERROR: {
        "message": "내부 서버 오류",
        "recoverable": False
    },
    ErrorCode.OUT_OF_MEMORY: {
        "message": "메모리 부족",
        "recoverable": False
    },
    ErrorCode.RATE_LIMITED: {
        "message": "요청 한도 초과",
        "recoverable": True
    },
}


class AIServerException(Exception):
    """AI 서버 커스텀 예외"""
    
    def __init__(self, code: ErrorCode, detail: str = None):
        self.code = code
        definition = ERROR_DEFINITIONS.get(code, {})
        self.message = definition.get("message", "알 수 없는 오류")
        self.detail = detail
        self.recoverable = definition.get("recoverable", False)
        self.error = ErrorDetail(
            code=code.value,
            message=self.message,
            detail=detail,
            recoverable=self.recoverable
        )
        super().__init__(f"[{code.value}] {self.message}: {detail}")


def create_error(code: ErrorCode, detail: str = None, fallback_used: bool = False) -> ErrorDetail:
    """에러 객체 생성 헬퍼"""
    definition = ERROR_DEFINITIONS.get(code, {})
    return ErrorDetail(
        code=code.value,
        message=definition.get("message", "알 수 없는 오류"),
        detail=detail,
        recoverable=definition.get("recoverable", True),
        fallback_used=fallback_used
    )


def get_fallback_feedback(category: str = "default") -> str:
    """폴백 피드백 메시지"""
    fallback_messages = {
        "정서적_고갈": "많이 지치셨을 것 같아요. 오늘은 푹 쉬세요.",
        "좌절_압박": "힘든 하루였군요. 잠시 숨 고르는 시간을 가져보세요.",
        "부정적_대인관계": "관계에서 오는 스트레스는 정말 힘들죠. 수고하셨어요.",
        "자기비하": "너무 자책하지 마세요. 당신은 충분히 잘하고 있어요.",
        "긍정": "오늘 하루도 수고하셨어요!",
        "default": "오늘 하루도 수고 많으셨어요. 잠시 쉬어가세요."
    }
    return fallback_messages.get(category, fallback_messages["default"])


if __name__ == "__main__":
    # 테스트
    print("에러 코드 목록:")
    for code in ErrorCode:
        definition = ERROR_DEFINITIONS.get(code, {})
        print(f"  {code.value}: {definition.get('message', 'N/A')} (복구가능: {definition.get('recoverable', False)})")
