# generate_network_viz.py
import json
import os
import networkx as nx
import matplotlib
matplotlib.use('Agg') # <<< GitHub Actions 같은 Non-GUI 환경에서 필수
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random
import math

# --- 설정값 ---
INPUT_FILE = 'data/grouped_keywords.json' # 그룹화된 키워드 JSON 파일
FONT_PATH = 'fonts/NanumGothic.ttf'      # 한글 폰트 경로 (필수!)
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'network_viz.png')

# --- 시각화 조정 변수 (여기서 크기/간격 조절) ---
NUM_COLS = 3         # 한 행에 표시할 그래프 수
NODE_SIZE_BASE = 3000 # 노드 기본 크기 (키움)
FONT_SIZE = 10      # 노드 레이블 폰트 크기 (조절)
LAYOUT_K = 0.8      # 노드 간 거리 조절 (클수록 멀어짐, 0.1 ~ 1.0+ 값으로 조절)
FIG_SCALE = 6       # 전체 이미지 크기 조절 (가로: NUM_COLS * FIG_SCALE, 세로: NUM_ROWS * FIG_SCALE)

EDGE_WIDTH = 0.6
NODE_ALPHA = 0.9
EDGE_ALPHA = 0.2

# 카테고리별 색상 지정 (하위 카테고리 기준)
SUBCATEGORY_COLORS = {
    # 디스플레이 산업
    '기술/소재/SCM': 'skyblue',
    '고객': 'lightcoral',
    '경쟁사': 'lightgreen',
    # IT 산업 (JSON 파일의 실제 하위 카테고리 이름에 맞춰야 함)
    # 예시: (실제 JSON 파일 내용에 따라 키 이름 변경 필요)
    '주요 동향 및 이슈': 'gold',
    'AI 및 소프트웨어': 'violet',
    '반도체 및 부품': 'lightsalmon',
    # 기본 색상 (위에서 못 찾을 경우)
    'default': 'grey'
}

def load_grouped_keywords(filepath):
    """JSON 파일에서 그룹화된 키워드 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"그룹화된 키워드 파일 로드 완료: {filepath}")
        # 하위 카테고리 개수 계산
        sub_category_count = sum(len(sub_cats) for sub_cats in data.values())
        print(f"총 하위 카테고리 개수: {sub_category_count}")
        return data, sub_category_count
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{filepath}'을 찾을 수 없습니다.")
        raise
    except json.JSONDecodeError:
        print(f"오류: '{filepath}' 파일이 유효한 JSON 형식이 아닙니다.")
        raise
    except Exception as e:
        print(f"파일 읽기/파싱 중 오류 발생: {e}")
        raise

def create_and_draw_subplots(grouped_data, num_sub_categories):
    """하위 카테고리별 네트워크 그래프를 서브플롯에 그리기"""

    # --- 한글 폰트 설정 ---
    try:
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"Font file not found: {FONT_PATH}")
        # Matplotlib의 전역 폰트 설정 변경
        font_name = fm.FontProperties(fname=FONT_PATH).get_name()
        fm.fontManager.addfont(FONT_PATH)
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False)
        print(f"폰트 설정 완료: {FONT_PATH} (Family: {font_name})")
    except Exception as e:
        print(f"오류: 폰트 파일을 설정할 수 없습니다. '{FONT_PATH}'. 오류: {e}")
        raise FileNotFoundError(f"Font file not found or invalid: {FONT_PATH}")
    # --- 폰트 설정 끝 ---

    # 서브플롯 행/열 계산
    num_rows = math.ceil(num_sub_categories / NUM_COLS)
    fig, axes = plt.subplots(num_rows, NUM_COLS, figsize=(NUM_COLS * FIG_SCALE, num_rows * FIG_SCALE))
    axes = axes.flatten() # 1차원 배열로 만들어 인덱싱 쉽게

    print(f"서브플롯 생성: {num_rows}행 x {NUM_COLS}열")

    plot_index = 0
    for main_category, sub_categories in grouped_data.items():
        for sub_category, keywords in sub_categories.items():
            if not keywords:
                print(f"Skipping empty sub-category: {sub_category}")
                # 빈 서브플롯 처리 (옵션)
                if plot_index < len(axes):
                    ax = axes[plot_index]
                    ax.set_title(f"{sub_category} (키워드 없음)", fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2))
                    ax.axis('off')
                    plot_index += 1
                continue

            if plot_index >= len(axes):
                print("경고: 서브플롯 개수보다 하위 카테고리가 많습니다. 일부는 그려지지 않습니다.")
                break

            print(f"'{sub_category}' 그래프 생성 및 그리기 시작...")
            # 1. 해당 하위 카테고리만의 그래프 생성
            G_sub = nx.Graph()
            G_sub.add_nodes_from(keywords)

            # 2. 하위 카테고리 내 노드끼리 연결 (Fully connected)
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    G_sub.add_edge(keywords[i], keywords[j])

            # 3. 현재 서브플롯 선택
            ax = axes[plot_index]

            # 4. 레이아웃 계산 (그래프가 작으므로 k값 조절 중요)
            pos = nx.spring_layout(G_sub, k=LAYOUT_K / math.sqrt(G_sub.number_of_nodes()) if G_sub.number_of_nodes() > 0 else LAYOUT_K, iterations=50, seed=42)

            # 5. 노드 색상 가져오기
            node_color = SUBCATEGORY_COLORS.get(sub_category, SUBCATEGORY_COLORS.get('default'))

            # 6. 그리기
            nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=NODE_SIZE_BASE, node_color=node_color, alpha=NODE_ALPHA)
            nx.draw_networkx_edges(G_sub, pos, ax=ax, width=EDGE_WIDTH, alpha=EDGE_ALPHA, edge_color='grey')
            nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=FONT_SIZE) # 전역 폰트 적용됨

            # 7. 서브플롯 제목 설정 및 축 숨기기
            ax.set_title(sub_category, fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2)) # 제목용 폰트
            ax.axis('off')
            print(f"'{sub_category}' 그래프 그리기 완료.")
            plot_index += 1

    # 남는 빈 서브플롯 숨기기
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    # 전체 그림 레이아웃 조정 및 저장
    plt.tight_layout(pad=2.0) # 서브플롯 간 간격 조절
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
        print(f"전체 네트워크 시각화 이미지가 '{OUTPUT_FILE}'로 저장되었습니다.")
    except Exception as e:
        print(f"ERROR saving image file: {e}")
        raise e
    finally:
        plt.close(fig) # Figure 객체 닫기

def main():
    """메인 실행 함수"""
    print(f"키워드 입력 파일: {INPUT_FILE}")
    try:
        grouped_data, num_sub_categories = load_grouped_keywords(INPUT_FILE)
        if grouped_data and num_sub_categories > 0:
            create_and_draw_subplots(grouped_data, num_sub_categories)
        else:
            print("오류: 입력 파일에서 유효한 그룹 데이터를 읽지 못했거나 하위 카테고리가 없습니다.")
    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
