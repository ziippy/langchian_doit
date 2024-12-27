import matplotlib.pyplot as plt

def plot_chart(input_data):
    # 딕셔너리 데이터 확인
    months = input_data.get("months", [])
    costs = input_data.get("costs", [])

    # 데이터 유효성 검사
    if not months or not costs:
        return "데이터가 올바르지 않습니다. 'months'와 'costs' 리스트가 필요합니다."
    if len(months) != len(costs):
        return "데이터 길이가 맞지 않습니다. 'months'와 'costs'의 항목 수가 같아야 합니다."

    # 차트 생성
    plt.figure(figsize=(10, 6))
    plt.plot(months, costs, marker='o', linestyle='-', linewidth=2)
    plt.title('Monthly Cloud Costs', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Cost ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(visible=True, linestyle='--', alpha=0.7)

    # 파일 저장
    file_name = "chart.png"
    plt.savefig(file_name)
    plt.close()
    return f"차트를 성공적으로 생성했습니다. 저장된 파일: {file_name}"
