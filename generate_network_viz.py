# generate_network_viz.py
import json
import os
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Non-GUI 환경용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
import numpy as np
import matplotlib.colors as mcolors

# --- 설정값 ---
INPUT_FILE = 'data/grouped_keywords.json' # 기존 JSON 파일 이름 사용
FONT_PATH = 'fonts/NanumGothic.ttf'      # 나눔고딕 폰트 경로 (필수!)
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'network_viz.png') # 기존 출력 파일 이름 사용

# --- 시각화 조정 변수 (2x2 레이아웃 기준) ---
NUM_COLS = 2         # 2x2 레이아웃
MAX_NODE_SIZE = 3200 # 최대 노드 크기
MIN_NODE_SIZE = 900  # 최소 노드 크기
MAX_NODE_ALPHA = 1.0 # 최대 투명도 (불투명)
MIN_NODE_ALPHA = 0.4 # 최소 투명도
FONT_SIZE = 8        # 폰트 크기
FONT_WEIGHT = 'bold' # 폰트 굵기
LAYOUT_K = 1.0       # 노드 간 기본 거리 (값을 키우면 더 멀어짐)
LAYOUT_ITERATIONS = 150 # 레이아웃 계산 반복 횟수
FIG_WIDTH = 10       # 전체 이미지 가로 크기
FIG_HEIGHT = 7       # 전체 이미지 세로 크기 (직사각형)

EDGE_WIDTH = 0.5     # 엣지 두께
EDGE_ALPHA = 0.15    # 엣지 투명도

# --- 사용할 컬러맵 ---
COLOR_MAP_NAME = 'OrRd' # <<< OrRd 컬러맵 사용
DEFAULT_COLOR = 'grey' # 예외 상황용

def load_grouped_keywords(filepath):
    """JSON 파일 로드 (단일 최상위 카테고리 아래 4개 하위 카테고리 구조 가정)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"그룹화된 키워드 파일 로드 완료: {filepath}")

        if len(data) == 1:
            main_key = list(data.keys())[0]
            sub_category_data = data[main_key]
            sub_category_count = len(sub_category_data)
            print(f"총 하위 카테고리 개수: {sub_category_count} (in '{main_key}')")
            if sub_category_count != NUM_COLS * 2: # 4개인지 확인
                 print(f"경고: 하위 카테고리 개수({sub_category_count})가 4개가 아닙니다. 레이아웃/색상 오류 가능성.")
            return sub_category_data
        else:
            print("오류: JSON 파일에 예상된 단일 최상위 카테고리가 없습니다.")
            raise ValueError("JSON format error: Expected a single top-level key.")

    except FileNotFoundError:
        print(f"오류: 입력 파일 '{filepath}'을 찾을 수 없습니다.")
        raise
    except json.JSONDecodeError:
        print(f"오류: '{filepath}' 파일이 유효한 JSON 형식이 아닙니다.")
        raise
    except Exception as e:
        print(f"파일 읽기/파싱 중 오류 발생: {e}")
        raise

def create_and_draw_subplots(sub_category_data):
    """하위 카테고리별 네트워크 그래프 그리기 (2x2, OrRd 컬러맵, 중요도 반영)"""

    num_sub_categories = len(sub_category_data)
    num_rows = math.ceil(num_sub_categories / NUM_COLS)

    # --- 한글 폰트 설정 ---
    try:
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"Font file not found: {FONT_PATH}")
        font_prop = fm.FontProperties(fname=FONT_PATH)
        font_name = font_prop.get_name()
        fm.fontManager.addfont(FONT_PATH)
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False)
        print(f"폰트 설정 완료: {FONT_PATH} (Family: {font_name})")
    except Exception as e:
        print(f"오류: 폰트 파일을 설정할 수 없습니다. '{FONT_PATH}'. 오류: {e}")
        raise FileNotFoundError(f"Font file not found or invalid: {FONT_PATH}")
    # --- 폰트 설정 끝 ---

    fig, axes = plt.subplots(num_rows, NUM_COLS, figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes = axes.flatten()

    print(f"서브플롯 생성: {num_rows}행 x {NUM_COLS}열")

    # --- 컬러맵 가져오기 ---
    cmap = plt.get_cmap(COLOR_MAP_NAME)

    plot_index = 0
    for sub_category, keywords in sub_category_data.items():
        if plot_index >= len(axes):
            print("경고: 서브플롯 개수보다 하위 카테고리가 많습니다.")
            break

        ax = axes[plot_index]

        if not keywords:
            print(f"Skipping empty sub-category: {sub_category}")
            ax.set_title(f"{sub_category} (키워드 없음)", fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2, weight=FONT_WEIGHT))
            ax.axis('off')
            plot_index += 1
            continue

        print(f"'{sub_category}' 그래프 생성 및 그리기 시작 (컬러맵: {COLOR_MAP_NAME})...")

        # 1. 그래프 생성 및 노드 추가
        G_sub = nx.Graph()
        center_node = None
        if keywords:
            center_node = keywords[0] # 가장 중요한 노드
            for keyword in keywords:
                G_sub.add_node(keyword)

        # 2. 엣지 추가
        keyword_list_for_edges = list(keywords)
        for i in range(len(keyword_list_for_edges)):
            for j in range(i + 1, len(keyword_list_for_edges)):
                G_sub.add_edge(keyword_list_for_edges[i], keyword_list_for_edges[j])

        # 3. 레이아웃 계산 (중앙 노드 고정, k값 조정)
        effective_k = LAYOUT_K
        num_nodes = G_sub.number_of_nodes()
        if num_nodes > 1:
             # 노드 수 증가에 따라 k값을 더 빠르게 감소시켜 봄 (예: 0.6 제곱근)
             effective_k = LAYOUT_K / (num_nodes ** 0.6)
             effective_k = max(0.05, effective_k) # 최소 k값 보장

        fixed_positions = {}
        initial_pos = {}
        fixed_nodes = []
        if center_node and center_node in G_sub:
            fixed_positions[center_node] = (0, 0) # 중앙 고정
            initial_pos[center_node] = (0, 0)
            fixed_nodes = [center_node]

        pos = nx.spring_layout(G_sub, k=effective_k,
                               pos=initial_pos if initial_pos else None,
                               fixed=fixed_nodes if fixed_nodes else None,
                               iterations=LAYOUT_ITERATIONS,
                               seed=42)

        # 4. 노드별 크기, 알파, 색상 계산
        keyword_to_size = {}
        keyword_to_alpha = {}
        keyword_to_color = {}
        num_keywords = len(keywords)

        for kw_idx, keyword in enumerate(keywords):
            # 중요도 비율 (0: 가장 중요, 1: 가장 덜 중요)
            importance_ratio = (kw_idx / (num_keywords - 1)) if num_keywords > 1 else 0

            # 크기/알파는 중요할수록 크게/불투명하게 (1 - 비율 사용)
            current_size = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * (1 - importance_ratio)
            keyword_to_size[keyword] = current_size
            current_alpha = MIN_NODE_ALPHA + (MAX_NODE_ALPHA - MIN_NODE_ALPHA) * (1 - importance_ratio)
            keyword_to_alpha[keyword] = current_alpha

            # OrRd 컬러맵 적용 및 그라데이션 방향 반전 (중요할수록 진한 빨강)
            color_ratio = 1.0 - importance_ratio
            current_color_rgba = cmap(color_ratio)
            keyword_to_color[keyword] = current_color_rgba

        # NetworkX draw 함수용 리스트 준비
        node_list = list(G_sub.nodes())
        ordered_sizes = [keyword_to_size.get(node, MIN_NODE_SIZE) for node in node_list]
        # ordered_alphas = [keyword_to_alpha.get(node, MIN_NODE_ALPHA) for node in node_list] # 개별 알파 적용 시 필요
        ordered_colors = [keyword_to_color.get(node, mcolors.to_rgb(DEFAULT_COLOR)) for node in node_list]

        # 5. 그리기
        nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=ordered_sizes, node_color=ordered_colors, alpha=MAX_NODE_ALPHA) # 노드 알파는 고정
        nx.draw_networkx_edges(G_sub, pos, ax=ax, width=EDGE_WIDTH, alpha=EDGE_ALPHA, edge_color='grey')
        nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=FONT_SIZE, font_family=font_name, font_weight=FONT_WEIGHT) # 굵은 글씨

        # 6. 서브플롯 제목 설정 (굵게) 및 축 숨기기
        ax.set_title(sub_category, fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2, weight=FONT_WEIGHT))
        ax.axis('off')
        print(f"'{sub_category}' 그래프 그리기 완료.")
        plot_index += 1

    # 남는 빈 서브플롯 숨기기 (4개 초과 시)
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    # 전체 그림 레이아웃 조정 및 저장
    plt.tight_layout(pad=4.0) # 여백 늘려서 잘림 방지
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white') # 배경 흰색
        print(f"전체 네트워크 시각화 이미지가 '{OUTPUT_FILE}'로 저장되었습니다.")
    except Exception as e:
        print(f"ERROR saving image file: {e}")
        raise e
    finally:
        plt.close(fig) # 메모리 해제

def main():
    """메인 실행 함수"""
    print(f"키워드 입력 파일: {INPUT_FILE}")
    try:
        sub_category_data = load_grouped_keywords(INPUT_FILE)
        if sub_category_data:
            create_and_draw_subplots(sub_category_data)
        else:
            print("오류: 입력 파일에서 유효한 그룹 데이터를 읽지 못했습니다.")
    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
