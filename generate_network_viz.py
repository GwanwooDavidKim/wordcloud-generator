# generate_network_viz.py
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

# --- 설정값 ---
INPUT_FILE = 'data/grouped_keywords.json' # 그룹화된 키워드 JSON 파일
FONT_PATH = 'fonts/NanumGothic.ttf'      # 한글 폰트 경로
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'network_viz.png')
FIG_SIZE = (18, 12) # 전체 이미지 크기 (인치 단위)
NODE_SIZE_BASE = 500 # 노드 기본 크기
FONT_SIZE = 10      # 노드 레이블 폰트 크기
EDGE_WIDTH = 0.5    # 엣지(선) 두께
LAYOUT_K = 0.5      # 노드 간 거리 조절 (클수록 멀어짐) - 조정 필요
NODE_ALPHA = 0.8    # 노드 투명도
EDGE_ALPHA = 0.3    # 엣지 투명도

# 카테고리별 색상 지정 (필요시 추가/변경)
CATEGORY_COLORS = {
    '디스플레이 산업': 'skyblue',
    'IT 산업': 'lightcoral',
    # 하위 카테고리별로 다른 색을 원하면 더 상세하게 지정 가능
}

def load_grouped_keywords(filepath):
    """JSON 파일에서 그룹화된 키워드 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("그룹화된 키워드 파일 로드 완료.")
        return data
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{filepath}'을 찾을 수 없습니다.")
        raise
    except json.JSONDecodeError:
        print(f"오류: '{filepath}' 파일이 유효한 JSON 형식이 아닙니다.")
        raise
    except Exception as e:
        print(f"파일 읽기/파싱 중 오류 발생: {e}")
        raise

def create_network_graph(grouped_data):
    """그룹화된 데이터를 바탕으로 NetworkX 그래프 생성"""
    G = nx.Graph()
    node_attributes = {}

    print("네트워크 그래프 생성 시작...")
    for main_category, sub_categories in grouped_data.items():
        main_cat_color = CATEGORY_COLORS.get(main_category, '#%06X' % random.randint(0, 0xFFFFFF)) # 기본 랜덤 색상

        for sub_category, keywords in sub_categories.items():
            if not keywords:
                continue

            # 중심 노드 추가 (하위 카테고리 이름) - 선택 사항
            # center_node = f"{sub_category}"
            # G.add_node(center_node)
            # node_attributes[center_node] = {'category': main_category, 'sub_category': sub_category, 'color': main_cat_color, 'is_center': True}


            # 키워드 노드 추가 및 중심 노드와 연결 (Method 1: 하위 카테고리 내 연결)
            # 모든 키워드를 해당 하위 카테고리의 중심 노드와 연결하거나, 키워드끼리 모두 연결
            nodes_in_subcategory = []
            for keyword in keywords:
                G.add_node(keyword)
                node_attributes[keyword] = {'category': main_category, 'sub_category': sub_category, 'color': main_cat_color, 'is_center': False}
                nodes_in_subcategory.append(keyword)
                # G.add_edge(center_node, keyword, weight=1) # 중심 노드와 연결 시

            # 하위 카테고리 내 키워드끼리 서로 연결 (밀집된 클러스터 형태)
            for i in range(len(nodes_in_subcategory)):
                for j in range(i + 1, len(nodes_in_subcategory)):
                    G.add_edge(nodes_in_subcategory[i], nodes_in_subcategory[j], weight=0.5) # weight는 예시

    nx.set_node_attributes(G, node_attributes)
    print(f"네트워크 그래프 생성 완료 (노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()})")
    return G

def draw_network(G):
    """네트워크 그래프 시각화 및 저장"""
    if not G.nodes():
        print("경고: 그래프에 노드가 없습니다. 이미지를 생성할 수 없습니다.")
        return

    # 한글 폰트 설정
    try:
        font_prop = fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE)
        print(f"폰트 로드 완료: {FONT_PATH}")
    except Exception as e:
        print(f"오류: 폰트 파일을 로드할 수 없습니다. '{FONT_PATH}'. 기본 폰트를 사용합니다. 오류: {e}")
        # 기본 폰트로 대체하거나 에러 처리
        # GitHub Actions에서는 에러 발생 시키는 것이 좋음
        raise FileNotFoundError(f"Font file not found or invalid: {FONT_PATH}")


    print("네트워크 시각화 시작...")
    plt.figure(figsize=FIG_SIZE)

    # 노드 위치 계산 (spring_layout 사용, k 값으로 노드 간 거리 조절)
    pos = nx.spring_layout(G, k=LAYOUT_K, iterations=50, seed=42) # seed 고정으로 레이아웃 일관성 유지

    # 노드 색상 및 크기 설정
    node_colors = [data['color'] for node, data in G.nodes(data=True)]
    # 노드 크기 (예: 중심 노드는 더 크게 - is_center 속성 사용 시)
    node_sizes = [NODE_SIZE_BASE * 2 if data.get('is_center', False) else NODE_SIZE_BASE for node, data in G.nodes(data=True)]

    # 엣지 가중치에 따른 두께 설정 (옵션)
    # edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    # nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=EDGE_ALPHA, edge_color='grey')
    nx.draw_networkx_edges(G, pos, width=EDGE_WIDTH, alpha=EDGE_ALPHA, edge_color='grey')

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=NODE_ALPHA)

    # 노드 레이블(키워드) 그리기
    nx.draw_networkx_labels(G, pos, font_prop=font_prop) # fontproperties 대신 font_prop 사용

    plt.title("Keyword Network Visualization", fontsize=16)
    plt.axis('off') # 축 숨기기
    plt.tight_layout()

    # 이미지 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150) # 해상도 조절 가능
    print(f"네트워크 시각화 이미지가 '{OUTPUT_FILE}'로 저장되었습니다.")
    plt.close()

def main():
    """메인 실행 함수"""
    print(f"키워드 입력 파일: {INPUT_FILE}")
    try:
        grouped_data = load_grouped_keywords(INPUT_FILE)
        if grouped_data:
            graph = create_network_graph(grouped_data)
            draw_network(graph)
        else:
            print("오류: 입력 파일에서 유효한 그룹 데이터를 읽지 못했습니다.")
    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
