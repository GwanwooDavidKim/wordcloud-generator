# generate_network_viz.py
import json
import os
import networkx as nx
import matplotlib
matplotlib.use('Agg') # <<< GitHub Actions 같은 Non-GUI 환경에서 필수
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
import numpy as np # 크기/알파 계산 위해 추가

# --- 설정값 ---
INPUT_FILE = 'data/grouped_keywords.json' # 그룹화된 키워드 JSON 파일 (중요도 순 정렬)
FONT_PATH = 'fonts/NanumGothic.ttf'      # 한글 폰트 경로 (필수!)
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'network_viz.png')

# --- 시각화 조정 변수 ---
NUM_COLS = 3         # 한 행에 표시할 그래프 수
MAX_NODE_SIZE = 3500 # 가장 중요한 노드 크기 (조금 키움)
MIN_NODE_SIZE = 800  # 가장 덜 중요한 노드 크기
MAX_NODE_ALPHA = 1.0 # 가장 중요한 노드 투명도 (불투명)
MIN_NODE_ALPHA = 0.3 # 가장 덜 중요한 노드 투명도
FONT_SIZE = 9       # 노드 레이블 폰트 크기 (조절)
LAYOUT_K = 0.9      # 노드 간 거리 조절 (클수록 멀어짐, 0.1 ~ 1.0+ 값으로 조절)
FIG_SCALE = 6       # 전체 이미지 크기 조절 (가로: NUM_COLS * FIG_SCALE, 세로: NUM_ROWS * FIG_SCALE)

EDGE_WIDTH = 0.5    # 엣지 두께 (조절)
EDGE_ALPHA = 0.15   # 엣지 투명도 (조절)

# --- 행별 색상 지정 ---
ROW1_COLOR = 'skyblue'  # 1행 그래프 색상 (파랑 계열)
ROW2_COLOR = 'lightcoral' # 2행 그래프 색상 (빨강 계열)
DEFAULT_COLOR = 'grey'  # 혹시 모를 기본 색상

def load_grouped_keywords(filepath):
    """JSON 파일에서 그룹화된 키워드 로드 (중요도 순으로 정렬된 리스트 기대)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"그룹화된 키워드 파일 로드 완료: {filepath}")
        # 하위 카테고리 개수 계산
        sub_category_count = sum(len(sub_cats) for sub_cats in data.values())
        print(f"총 하위 카테고리 개수: {sub_category_count}")
        if sub_category_count != NUM_COLS * 2: # 2행 3열 구조 확인
             print(f"경고: 하위 카테고리 개수({sub_category_count})가 6개가 아닙니다. 레이아웃이 예상과 다를 수 있습니다.")
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
    """하위 카테고리별 네트워크 그래프를 서브플롯에 그리기 (중요도 반영)"""

    # --- 한글 폰트 설정 ---
    try:
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"Font file not found: {FONT_PATH}")
        font_name = fm.FontProperties(fname=FONT_PATH).get_name()
        fm.fontManager.addfont(FONT_PATH)
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False)
        print(f"폰트 설정 완료: {FONT_PATH} (Family: {font_name})")
    except Exception as e:
        print(f"오류: 폰트 파일을 설정할 수 없습니다. '{FONT_PATH}'. 오류: {e}")
        raise FileNotFoundError(f"Font file not found or invalid: {FONT_PATH}")
    # --- 폰트 설정 끝 ---

    # 서브플롯 행/열 계산 (2행 3열 고정 가정)
    num_rows = math.ceil(num_sub_categories / NUM_COLS)
    if num_rows != 2 or num_sub_categories > 6:
        print("경고: 2행 3열(총 6개) 레이아웃을 벗어납니다. 색상 및 배치가 이상할 수 있습니다.")
        # 필요시 num_rows 강제 조정 또는 오류 처리
        num_rows = max(2, num_rows) # 최소 2행 확보 시도

    fig, axes = plt.subplots(num_rows, NUM_COLS, figsize=(NUM_COLS * FIG_SCALE, num_rows * FIG_SCALE))
    axes = axes.flatten() # 1차원 배열로 만들어 인덱싱 쉽게

    print(f"서브플롯 생성: {num_rows}행 x {NUM_COLS}열")

    plot_index = 0
    # grouped_data의 순서가 중요 (JSON 파일의 카테고리 순서대로 그려짐)
    for main_category, sub_categories in grouped_data.items():
        for sub_category, keywords in sub_categories.items():
            if plot_index >= len(axes):
                print("경고: 서브플롯 개수보다 하위 카테고리가 많습니다. 일부는 그려지지 않습니다.")
                break

            ax = axes[plot_index] # 현재 서브플롯

            # 행에 따른 색상 결정
            if plot_index < NUM_COLS: # 1행 (0, 1, 2)
                current_color = ROW1_COLOR
            elif plot_index < NUM_COLS * 2: # 2행 (3, 4, 5)
                current_color = ROW2_COLOR
            else: # 6개를 넘어가는 경우 (있으면 안되지만 대비)
                current_color = DEFAULT_COLOR

            if not keywords:
                print(f"Skipping empty sub-category: {sub_category}")
                ax.set_title(f"{sub_category} (키워드 없음)", fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2))
                ax.axis('off')
                plot_index += 1
                continue

            print(f"'{sub_category}' 그래프 생성 및 그리기 시작 (색상: {current_color})...")

            # 1. 그래프 생성 및 노드 추가 (순서 유지를 위해 keywords 리스트 사용)
            G_sub = nx.Graph()
            # 노드를 keywords 리스트 순서대로 추가 (NetworkX 2.x 이상에서는 순서가 유지될 수 있음)
            for keyword in keywords:
                 G_sub.add_node(keyword)

            # 2. 하위 카테고리 내 노드끼리 연결 (Fully connected)
            keyword_list_for_edges = list(keywords) # 명시적으로 리스트화
            for i in range(len(keyword_list_for_edges)):
                for j in range(i + 1, len(keyword_list_for_edges)):
                    G_sub.add_edge(keyword_list_for_edges[i], keyword_list_for_edges[j])

            # 3. 레이아웃 계산
            # 노드 수가 적을수록 k값을 키워 넓게 퍼지도록 조정 (기존 로직 개선)
            effective_k = LAYOUT_K
            if G_sub.number_of_nodes() > 1:
                 effective_k = LAYOUT_K / np.log(G_sub.number_of_nodes() + 1) # 로그 스케일로 조정 시도
                 effective_k = max(0.1, effective_k) # 최소값 보장

            pos = nx.spring_layout(G_sub, k=effective_k, iterations=100, seed=42) # iteration 증가

            # 4. 노드별 크기 및 투명도 계산 (중요도 순서 반영)
            node_sizes = []
            node_alphas = []
            num_keywords = len(keywords)

            for kw_idx, keyword in enumerate(keywords):
                # 인덱스(중요도 순위)에 따라 크기와 알파값을 선형 또는 비선형적으로 감소시킴
                # 예: 선형 감소
                size_ratio = max(0, (num_keywords - 1 - kw_idx) / (num_keywords - 1)) if num_keywords > 1 else 1
                current_size = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * size_ratio

                alpha_ratio = max(0, (num_keywords - 1 - kw_idx) / (num_keywords - 1)) if num_keywords > 1 else 1
                current_alpha = MIN_NODE_ALPHA + (MAX_NODE_ALPHA - MIN_NODE_ALPHA) * alpha_ratio

                # # 예: 지수적 감소 (더 급격히 작아짐)
                # decay_rate_size = 0.8
                # decay_rate_alpha = 0.75
                # current_size = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * (decay_rate_size ** kw_idx)
                # current_alpha = MIN_NODE_ALPHA + (MAX_NODE_ALPHA - MIN_NODE_ALPHA) * (decay_rate_alpha ** kw_idx)

                node_sizes.append(current_size)
                node_alphas.append(current_alpha)

            # NetworkX draw 함수는 node list 순서에 맞는 size/alpha list를 기대함
            # G_sub.nodes() 순서와 keywords 리스트 순서가 다를 수 있으므로 매핑 필요
            node_list = list(G_sub.nodes())
            size_map = {kw: sz for kw, sz in zip(keywords, node_sizes)}
            alpha_map = {kw: al for kw, al in zip(keywords, node_alphas)}

            ordered_sizes = [size_map.get(node, MIN_NODE_SIZE) for node in node_list]
            ordered_alphas = [alpha_map.get(node, MIN_NODE_ALPHA) for node in node_list]


            # 5. 그리기 (노드 크기/알파 리스트 사용)
            nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=ordered_sizes, node_color=current_color, alpha=MAX_NODE_ALPHA) # 알파는 여기서 전체적으로 한번 주고, 개별 노드 투명도는 레이블에서 조절하거나, 아래처럼 개별 노드에 적용
            # nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=ordered_sizes, node_color=current_color, alpha=ordered_alphas) # 노드 자체에 알파 적용

            nx.draw_networkx_edges(G_sub, pos, ax=ax, width=EDGE_WIDTH, alpha=EDGE_ALPHA, edge_color='grey')

            # 레이블 폰트 크기는 고정 또는 노드 크기에 비례하게 조절 가능
            # 여기서는 고정 FONT_SIZE 사용
            nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=FONT_SIZE, font_family=font_name) # font_family 명시

            # 6. 서브플롯 제목 설정 및 축 숨기기
            ax.set_title(sub_category, fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2))
            ax.axis('off')
            print(f"'{sub_category}' 그래프 그리기 완료.")
            plot_index += 1

    # 남는 빈 서브플롯 숨기기
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    # 전체 그림 레이아웃 조정 및 저장
    plt.tight_layout(pad=3.0) # 서브플롯 간 간격 조절 (조금 더 넓힘)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white') # 배경색 흰색 지정
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
        # JSON 파일 로드 시 grouped_keywords.json 으로 경로 수정 필요
        grouped_data, num_sub_categories = load_grouped_keywords(INPUT_FILE)
        if grouped_data and num_sub_categories > 0:
            create_and_draw_subplots(grouped_data, num_sub_categories)
        else:
            print("오류: 입력 파일에서 유효한 그룹 데이터를 읽지 못했거나 하위 카테고리가 없습니다.")
    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
