# generate_network_viz.py
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random
import numpy as np  # Pillow과 함께 마스크 사용 시 필요할 수 있음
from PIL import Image # 마스크 사용 시 필요할 수 있음

# --- 설정값 ---
INPUT_FILE = 'data/grouped_keywords.json' # 그룹화된 키워드 JSON 파일 경로
FONT_PATH = 'fonts/NanumGothic.ttf'      # 레포지토리 내 한글 폰트 파일 경로 (예: 나눔고딕)
OUTPUT_DIR = 'output'                     # 출력 이미지를 저장할 디렉토리
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'network_viz.png') # 최종 저장될 이미지 파일 경로
FIG_SIZE = (20, 15) # 전체 이미지 크기 (인치 단위) - 필요시 조절
NODE_SIZE_BASE = 600 # 노드 기본 크기 - 필요시 조절
FONT_SIZE = 9       # 노드 레이블 폰트 크기 - 필요시 조절
EDGE_WIDTH = 0.5    # 엣지(선) 두께
LAYOUT_K = 0.6      # 노드 간 거리 조절 (클수록 멀어짐, 0.1 ~ 1.0 사이 값으로 조절해보세요)
NODE_ALPHA = 0.8    # 노드 투명도
EDGE_ALPHA = 0.3    # 엣지 투명도

# 카테고리별 색상 지정 (더 많은 카테고리나 다른 색상으로 변경 가능)
CATEGORY_COLORS = {
    '디스플레이 산업': 'skyblue',
    'IT 산업': 'lightcoral',
    # 필요하다면 하위 카테고리별로 다른 색상 지정 가능
    # 예: '기술/소재/SCM': 'lightblue', '고객': 'deepskyblue', ...
}

def load_grouped_keywords(filepath):
    """JSON 파일에서 그룹화된 키워드 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"그룹화된 키워드 파일 로드 완료: {filepath}")
        return data
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{filepath}'을 찾을 수 없습니다.")
        raise # 워크플로우 실패 유도
    except json.JSONDecodeError:
        print(f"오류: '{filepath}' 파일이 유효한 JSON 형식이 아닙니다.")
        raise # 워크플로우 실패 유도
    except Exception as e:
        print(f"파일 읽기/파싱 중 오류 발생: {e}")
        raise

def create_network_graph(grouped_data):
    """그룹화된 데이터를 바탕으로 NetworkX 그래프 생성"""
    G = nx.Graph()
    node_attributes = {}

    print("네트워크 그래프 생성 시작...")
    color_index = 0
    predefined_colors = list(CATEGORY_COLORS.values())
    random_colors = ['#FFC0CB', '#90EE90', '#FFD700', '#E6E6FA', '#FFA07A'] # 추가 랜덤 색상 예시

    for main_category, sub_categories in grouped_data.items():
        # 메인 카테고리별 기본 색상 가져오기 또는 랜덤 할당
        main_cat_color = CATEGORY_COLORS.get(main_category)
        if main_cat_color is None:
            if color_index < len(predefined_colors):
                 main_cat_color = predefined_colors[color_index]
                 color_index += 1
            else:
                 main_cat_color = random.choice(random_colors) # 정의되지 않은 카테고리는 랜덤 색상

        for sub_category, keywords in sub_categories.items():
            if not keywords:
                print(f"경고: 하위 카테고리 '{sub_category}'에 키워드가 없습니다.")
                continue

            nodes_in_subcategory = []
            for keyword in keywords:
                # 노드가 이미 존재하면 속성 업데이트, 없으면 추가
                if keyword not in G:
                    G.add_node(keyword)
                    node_attributes[keyword] = {'category': main_category, 'sub_category': sub_category, 'color': main_cat_color}
                else: # 다른 카테고리에서 이미 추가된 경우 속성 업데이트 (예: 색상 등)
                    node_attributes[keyword]['sub_category'] += f", {sub_category}" # 여러 카테고리에 속할 경우 표시 (선택적)

                nodes_in_subcategory.append(keyword)

            # 하위 카테고리 내 키워드끼리 서로 연결 (밀집된 클러스터 형태)
            # 연결 수가 너무 많아지면 그래프가 복잡해질 수 있음
            for i in range(len(nodes_in_subcategory)):
                for j in range(i + 1, len(nodes_in_subcategory)):
                    # 이미 엣지가 있으면 추가하지 않음 (Graph는 중복 엣지 허용 안 함)
                    G.add_edge(nodes_in_subcategory[i], nodes_in_subcategory[j], weight=0.5) # weight는 시각화에 활용 가능

    nx.set_node_attributes(G, node_attributes)
    print(f"네트워크 그래프 생성 완료 (노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()})")
    return G

def draw_network(G):
    """네트워크 그래프 시각화 및 저장"""
    if not G.nodes():
        print("경고: 그래프에 노드가 없습니다. 이미지를 생성할 수 없습니다.")
        return

    # --- 한글 폰트 설정 (수정된 안정적인 방식) ---
    try:
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"Font file not found: {FONT_PATH}")

        # Matplotlib의 전역 폰트 설정 변경
        font_name = fm.FontProperties(fname=FONT_PATH).get_name()
        fm.fontManager.addfont(FONT_PATH) # 폰트 매니저에 폰트 추가
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False) # 마이너스 부호 깨짐 방지
        print(f"폰트 설정 완료: {FONT_PATH} (Family: {font_name})")

    except Exception as e:
        print(f"오류: 폰트 파일을 설정할 수 없습니다. '{FONT_PATH}'. 오류: {e}")
        raise FileNotFoundError(f"Font file not found or invalid: {FONT_PATH}")
    # --- 폰트 설정 끝 ---

    print("네트워크 시각화 시작...")
    plt.figure(figsize=FIG_SIZE)

    # 노드 위치 계산 (spring_layout 사용, k값으로 노드 간 거리 조절)
    # k값을 늘리면 노드들이 더 멀리 퍼지고, 줄이면 더 가까이 모입니다.
    # iterations 값을 늘리면 레이아웃 계산이 더 안정화될 수 있습니다.
    print(f"레이아웃 계산 시작 (k={LAYOUT_K})...")
    pos = nx.spring_layout(G, k=LAYOUT_K, iterations=50, seed=42)
    print("레이아웃 계산 완료.")

    # 노드 색상 및 크기 설정
    node_colors = [data.get('color', 'grey') for node, data in G.nodes(data=True)] # 색상 정보 가져오기 (없으면 회색)
    # 노드 크기는 기본 크기로 통일 (조정 가능)
    node_sizes = [NODE_SIZE_BASE for _ in G.nodes()]

    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, width=EDGE_WIDTH, alpha=EDGE_ALPHA, edge_color='grey')

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=NODE_ALPHA)

    # 노드 레이블(키워드) 그리기 (font_family는 전역 설정 사용)
    nx.draw_networkx_labels(G, pos, font_size=FONT_SIZE)

    plt.title("Keyword Network Visualization", fontsize=16) # 제목도 설정된 한글 폰트로 표시됨
    plt.axis('off') # 축 숨기기
    plt.tight_layout() # 레이아웃 여백 조절

    # 출력 디렉토리 생성 (없으면)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 이미지 저장
    try:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight') # bbox_inches='tight'로 여백 최소화 시도
        print(f"이미지 저장 시도 완료! 파일: '{OUTPUT_FILE}'")
    except Exception as e:
        print(f"ERROR saving image file: {e}")
        raise e
    finally:
         plt.close() # Figure 객체 닫아서 메모리 해제

def main():
    """메인 실행 함수"""
    print(f"키워드 입력 파일: {INPUT_FILE}")
    try:
        grouped_data = load_grouped_keywords(INPUT_FILE)
        if grouped_data:
            graph = create_network_graph(grouped_data)
            # 생성된 노드 및 엣지 정보 일부 출력 (디버깅용)
            print("\n--- 그래프 정보 일부 ---")
            print(f"Nodes: {list(graph.nodes())[:10]}...") # 처음 10개 노드
            print(f"Edges: {list(graph.edges())[:10]}...") # 처음 10개 엣지
            draw_network(graph)
        else:
            print("오류: 입력 파일에서 유효한 그룹 데이터를 읽지 못했습니다.")
    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
