from langchain.tools import tool

import json
from plot_chart import plot_chart

@tool
def chart_tool(input: str) -> str:
    """
    딕셔너리 형태의 데이터를 기반으로 차트를 생성하는 함수.
    Args:
        input_data (dict): {"months": [...], "costs": [...]}
    Returns:
        str: 차트 생성 결과 메시지.
    """
    try:
        # JSON 파싱
        input_data = json.loads(input)
        return plot_chart(input_data)
    except Exception as e:
        return f"오류 발생: {e}"
