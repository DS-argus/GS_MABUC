import itertools
from collections import defaultdict #키 값을 디폴트로 지정할 수 있음
from itertools import product

import functools
import networkx as nx
from pyvis.network import Network
import numpy as np
from typing import Dict, Iterable, Optional, Set, Sequence, AbstractSet
from typing import FrozenSet, Tuple

from utils import fzset_union, sortup, sortup2, with_default

import matplotlib.pyplot as plt

## mu에는 dict로 U가 1일 확률 넣어줌
# 이 함수의 return 값에 U의 realization을 담은 dict인 d를 넣어주면 mu에 있는 U를 돌면서 그 조합이 나올 확률 계산
def default_P_U(mu: Dict):
    """ P(U) function given a dictionary of probabilities for each U_i being 1, P(U_i=1) """

    def P_U(d):
        p_val = 1.0
        for k in mu.keys():
            p_val *= (1 - mu[k]) if d[k] == 0 else mu[k]
        return p_val

    return P_U


# keys에 있는 key들로 dict 가져오기
def dict_only(a_dict: dict, keys: AbstractSet) -> Dict:
    return {k: a_dict[k] for k in keys if k in a_dict}

# keys에 없는 key들로 dict 가져오기
def dict_except(a_dict: dict, keys: AbstractSet) -> Dict:
    return {k: v for k, v in a_dict.items() if k not in keys}

# (X, Y)가 들어오면 Y를 X의 children으로 바꿔줌
def pairs2dict(xys, backward=False):
    dd = defaultdict(set)
    if backward:
        for x, y in xys:
            dd[y].add(x)
    else:
        for x, y in xys:
            dd[x].add(y)

    return defaultdict(frozenset, {key: frozenset(vals) for key, vals in dd.items()})

# frozenset으로 wrapping
def wrap(v_or_vs, wrap_with=frozenset):
    if v_or_vs is None:
        return None
    if isinstance(v_or_vs, str):    ## string type이면 {}로 감싸서 frozenset으로
        return wrap_with({v_or_vs})
    else:                           ## v여러개를 list로 입력하면 모두 set으로 변환
        return wrap_with(v_or_vs)


class CausalDiagram:
    def __init__(self,
                 vs: Optional[Iterable[str]],
                 directed_edges: Optional[Iterable[Tuple[str, str]]] = frozenset(),
                 bidirected_edges: Optional[Iterable[Tuple[str, str, str]]] = frozenset(),
                 copy: 'CausalDiagram' = None,
                 with_do: Optional[Set[str]] = None,
                 with_induced: Optional[Set[str]] = None):
        with_do = wrap(with_do)
        with_induced = wrap(with_induced)

        # do, induced 메소드에서 활용됨
        if copy is not None:
            if with_do is not None: ##do 하는 변수에 대한 변경사항 기록
                self.V = copy.V
                self.U = wrap(u for u in copy.U if with_do.isdisjoint(copy.confounded_dict[u]))
                self.confounded_dict = {u: val for u, val in copy.confounded_dict.items() if u in self.U}

                # copy cautiously
                dopa = copy.pa(with_do)
                doAn = copy.An(with_do)
                doDe = copy.De(with_do)

                self._pa = defaultdict(frozenset, {k: frozenset() if k in with_do else v for k, v in copy._pa.items()})
                self._ch = defaultdict(frozenset, {k: (v - with_do) if k in dopa else v for k, v in copy._ch.items()})
                self._an = dict_except(copy._an, doDe)
                self._de = dict_except(copy._de, doAn)

            elif with_induced is not None:  # copy하고 induced에 포함된 V만 남기고 나머지는 제거
                assert with_induced <= copy.V
                removed = copy.V - with_induced
                self.V = with_induced
                self.confounded_dict = {u: val for u, val in copy.confounded_dict.items() if val <= self.V}
                self.U = wrap(self.confounded_dict)

                children_are_removed = copy.pa(removed) & self.V
                parents_are_removed = copy.ch(removed) & self.V
                ancestors_are_removed = copy.de(removed) & self.V
                descendants_are_removed = copy.an(removed) & self.V

                self._pa = defaultdict(frozenset, {x: (copy._pa[x] - removed) if x in parents_are_removed else copy._pa[x] for x in self.V})
                self._ch = defaultdict(frozenset, {x: (copy._ch[x] - removed) if x in children_are_removed else copy._ch[x] for x in self.V})
                self._an = dict_only(copy._an, self.V - ancestors_are_removed)
                self._de = dict_only(copy._de, self.V - descendants_are_removed)
            else: # 그냥 copy만 했을 때, 필요한 정보 복사
                self.V = copy.V
                self.U = copy.U
                self.confounded_dict = copy.confounded_dict
                self._ch = copy._ch
                self._pa = copy._pa
                self._an = copy._an
                self._de = copy._de
        else:
            directed_edges = list(directed_edges)
            bidirected_edges = list(bidirected_edges)
            self.V = frozenset(vs) | fzset_union(directed_edges) | fzset_union((x, y) for x, y, _ in bidirected_edges)
            self.U = frozenset(u for _, _, u in bidirected_edges)
            self.confounded_dict = {u: frozenset({x, y}) for x, y, u in
                                    bidirected_edges}

            self._ch = pairs2dict(directed_edges)
            self._pa = pairs2dict(directed_edges, backward=True)
            self._an = dict()  # cache
            self._de = dict()  # cache
            assert self._ch.keys() <= self.V and self._pa.keys() <= self.V
        
        self.bidirected_edges = list(bidirected_edges)
        self.edges = tuple((x, y) for x, ys in self._ch.items() for y in ys)

        # https://kimjingo.tistory.com/169 : lru_cache()
        self.causal_order = functools.lru_cache()(self.causal_order)
        self._do_ = functools.lru_cache()(self._do_)
        self.__cc = None
        self.__cc_dict = None
        self.__h = None
        self.__characteristic = None
        self.__confoundeds = None
        self.u_pas = defaultdict(set)
        for u, xy in self.confounded_dict.items():
            for v in xy:
                self.u_pas[v].add(u)
        # u_pas[v] : v와 연결된 uc를 갖고 있음        
        self.u_pas = defaultdict(set, {v: frozenset(us) for v, us in self.u_pas.items()})

    # v와 연결된 UC return
    def UCs(self, v):
        return self.u_pas[v]

    # in 연산자에 사용
    def __contains__(self, item):
        if isinstance(item, str):   # 'x' -> 노드 또는 edge에 포함되는지
            return item in self.V or item in self.U
        if len(item) == 2:          # (x, y) -> bidirected edge로 연결되어있는지
            if isinstance(item, AbstractSet):
                x, y = item
                return self.is_confounded(x, y)
            else:                   # edge로 연결되어 있는지
                return tuple(item) in self.edges
        if len(item) == 3:
            x, y, u = item          # x,y가 confounded되어있고 u가 이름이 맞는지, 그 u가 x,y를 연결하는지 
            return self.is_confounded(x, y) and u in self.confounded_dict and self.confounded_dict[u] == frozenset({x, y})
        return False

    # CD간 대소비교?
    def __lt__(self, other):
        if not isinstance(other, CausalDiagram):
            return False
        return self <= other and self != other
    
    def __le__(self, other):
        if not isinstance(other, CausalDiagram):
            return False
        return self.V <= other.V and set(self.edges) <= set(other.edges) and set(self.confounded_dict.values()) <= set(other.confounded_dict.values())

    def __ge__(self, other):
        if not isinstance(other, CausalDiagram):
            return False
        return self.V >= other.V and set(self.edges) >= set(other.edges) and set(self.confounded_dict.values()) >= set(other.confounded_dict.values())

    def __gt__(self, other):
        if not isinstance(other, CausalDiagram):
            return False
        return self >= other and self != other

    # 간단한 함수들
    def Pa(self, v_or_vs) -> FrozenSet:
        return self.pa(v_or_vs) | wrap(v_or_vs, frozenset)

    def pa(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self._pa[v_or_vs]
        else:
            return fzset_union(self._pa[v] for v in v_or_vs)

    def ch(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self._ch[v_or_vs]
        else:
            return fzset_union(self._ch[v] for v in v_or_vs)

    def Ch(self, v_or_vs) -> FrozenSet:
        return self.ch(v_or_vs) | wrap(v_or_vs, frozenset)

    def An(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self.__an(v_or_vs) | {v_or_vs}
        return self.an(v_or_vs) | wrap(v_or_vs, frozenset)

    def an(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self.__an(v_or_vs)
        return fzset_union(self.__an(v) for v in wrap(v_or_vs))

    # de구하고 parents포함  union
    def De(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self.__de(v_or_vs) | {v_or_vs}
        return self.de(v_or_vs) | wrap(v_or_vs, frozenset)

    # str로 입력받으면 __de사용, 다른 형태(list)면 wrap하고 하나하나 __de 구해서 union
    def de(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self.__de(v_or_vs)
        return fzset_union(self.__de(v) for v in wrap(v_or_vs))

    # v의 ancestor를 return
    def __an(self, v) -> FrozenSet:
        if v in self._an:
            return self._an[v]
        self._an[v] = fzset_union(self.__an(parent) for parent in self._pa[v]) | self._pa[v]
        return self._an[v]

    # v의 descendant를 return
    def __de(self, v) -> FrozenSet:
        if v in self._de:   ## caching 되어 있으면 그거 return
            return self._de[v]
        # recursive하게 child의 child를 계속 탐색하면서 결과를 chain으로 엮어서 frozenset으로 만들고 child와 union
        self._de[v] = fzset_union(self.__de(child) for child in self._ch[v]) | self._ch[v]
        return self._de[v]

    # do하는 변수들을 wrapping해서 _do_ 실행
    def do(self, v_or_vs) -> 'CausalDiagram':
        return self._do_(wrap(v_or_vs))

    # copy하고 do실행
    def _do_(self, v_or_vs) -> 'CausalDiagram':
        return CausalDiagram(None, None, None, self, wrap(v_or_vs))

    # edge 있는지 여부 return
    def has_edge(self, x, y) -> bool:
        return y in self._ch[x]

    # 두 노드가 UC로 연결되어 있으면 true
    def is_confounded(self, x, y) -> bool:
        return {x, y} in self.confounded_dict.values()

    # 두 노드에 대한 UC return
    def u_of(self, x, y):
        key = {x, y}
        for u, ab in self.confounded_dict.items():
            if ab == key:
                return u
        return None

    # UC의 이름을 넣으면 UC로 연결된 노드 return
    def confounded_with(self, u):
        return self.confounded_dict[u]

    # v와 confounded된 변수 return
    def confounded_withs(self, v):
        return {next(iter(xy - {v})) for xy in self.confounded_dict.values() if v in xy}

    # 실행시키면 induced된 CD return
    def __getitem__(self, item) -> 'CausalDiagram':
        return self.induced(item)

    # 일부 노드만으로 구성된 CD return?
    def induced(self, v_or_vs) -> 'CausalDiagram':
        if set(v_or_vs) == self.V:
            return self
        return CausalDiagram(None, None, None, copy=self, with_induced=v_or_vs)

    # CD 요약정보: 노드 개수 .edge 개수 , bidireced edge 개수, 노드별 ch, pa, UC 연결 개수
    @property
    def characteristic(self):
        if self.__characteristic is None:
            self.__characteristic = (len(self.V),
                                     len(self.edges),
                                     len(self.confounded_dict),
                                     sortup([(len(self.ch(v)), len(self.pa(v)), len(self.confounded_withs(v))) for v in self.V]))
        return self.__characteristic

    # edge를 제거한 diagram을 return
    def edges_removed(self, edges_to_remove: Iterable[Sequence[str]]) -> 'CausalDiagram':
        edges_to_remove = [tuple(edge) for edge in edges_to_remove]

        dir_edges = {edge for edge in edges_to_remove if len(edge) == 2}
        bidir_edges = {edge for edge in edges_to_remove if len(edge) == 3}
        bidir_edges = frozenset((*sorted([x, y]), u) for x, y, u in bidir_edges)
        return CausalDiagram(self.V, set(self.edges) - dir_edges, self.confounded_to_3tuples() - bidir_edges)

    # 일부 노드 혹은 edge를 제거하는 함수
    def __sub__(self, v_or_vs_or_edges) -> 'CausalDiagram':
        if not v_or_vs_or_edges:
            return self
        if isinstance(v_or_vs_or_edges, str):
            return self[self.V - wrap(v_or_vs_or_edges)]
        v_or_vs_or_edges = list(v_or_vs_or_edges)
        if isinstance(v_or_vs_or_edges[0], str):
            return self[self.V - wrap(v_or_vs_or_edges)]
        return self.edges_removed(v_or_vs_or_edges)

    # topological order 대로 tuple을 만들어주는 함수
    def causal_order(self, backward=False) -> Tuple:
        gg = nx.DiGraph(self.edges)
        gg.add_nodes_from(self.V)
        top_to_bottom = list(nx.topological_sort(gg))
        if backward:
            return tuple(reversed(top_to_bottom))
        else:
            return tuple(top_to_bottom)

    # CD에 다른 CD 혹은 edge들을 추가하는 함수
    def __add__(self, edges):
        if isinstance(edges, CausalDiagram):
            return merge_two_cds(self, edges)

        directed_edges = {edge for edge in edges if len(edge) == 2}
        bidirected_edges = {edge for edge in edges if len(edge) == 3}
        return CausalDiagram(self.V, set(self.edges) | directed_edges, self.confounded_to_3tuples() | bidirected_edges)

    # CC한번 구하면 __counfoundeds 저장
    def __ensure_confoundeds_cached(self):
        if self.__confoundeds is None:
            self.__confoundeds = dict()
            for u, (x, y) in self.confounded_dict.items():
                if x not in self.__confoundeds:
                    self.__confoundeds[x] = set()
                if y not in self.__confoundeds:
                    self.__confoundeds[y] = set()
                self.__confoundeds[x].add(y)
                self.__confoundeds[y].add(x)
            self.__confoundeds = {x: frozenset(ys) for x, ys in self.__confoundeds.items()}
            for v in self.V:
                if v not in self.__confoundeds:
                    self.__confoundeds[v] = frozenset()

    ## BFS 방식으로 CC 구하는 알고리즘
    ## CC한번 구하면 self.__cc, self.__cc_dict에 저장
    def __ensure_cc_cached(self):
        if self.__cc is None:
            self.__ensure_confoundeds_cached()
            ccs = []
            remain = set(self.V)
            found = set()
            while remain:   # 모든 노드를 다 돌아야 함
                v = next(iter(remain))
                a_cc = set()
                to_expand = [v]
                while to_expand:    # bidirected로 연결된 노드 모두 방문
                    v = to_expand.pop()
                    a_cc.add(v)
                    to_expand += list(self.__confoundeds[v] - a_cc)
                ccs.append(a_cc)    # CC list에 방금 구한 CC 추가
                found |= a_cc
                remain -= found
            self.__cc2 = frozenset(frozenset(a_cc) for a_cc in ccs)
            self.__cc_dict2 = {v: a_cc for a_cc in self.__cc2 for v in a_cc}

            self.__cc = self.__cc2
            self.__cc_dict = self.__cc_dict2

    # 모든 CC return
    @property
    def c_components(self) -> FrozenSet:
        self.__ensure_cc_cached()
        return self.__cc

    # 노드가 속하는 CC return
    def c_component(self, v_or_vs) -> FrozenSet:
        assert isinstance(v_or_vs, str)
        self.__ensure_cc_cached()
        return fzset_union(self.__cc_dict[v] for v in wrap(v_or_vs))

    # bidirected edges를 담은 dictionary를 (x, y, u) 형태로 변경
    def confounded_to_3tuples(self) -> FrozenSet[Tuple[str, str, str]]:
        return frozenset((*sorted([x, y]), u) for u, (x, y) in self.confounded_dict.items())

    def __eq__(self, other):    #V, edges ,confounder가 모두 같아야 동일한 causaldiagram
        if not isinstance(other, CausalDiagram):
            return False
        if self.V != other.V:
            return False
        if set(self.edges) != set(other.edges):
            return False
        if set(self.confounded_dict.values()) != set(other.confounded_dict.values()):  # does not care about U's name
            return False
        return True

    # https://codingdog.tistory.com/entry/python-hash-%ED%95%A8%EC%88%98%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B4%85%EC%8B%9C%EB%8B%A4
    def __hash__(self): # causaldiagram의 hash값 return하는 듯?
        if self.__h is None:
            self.__h = hash(sortup(self.V)) ^ hash(sortup(self.edges)) ^ hash(sortup2(self.confounded_dict.values()))
        return self.__h

    # https://shoark7.github.io/programming/python/difference-between-__repr__-vs-__str__
    def __repr__(self): # 출력형태
        return cd2qcd(self)

    def __str__(self):  # 출력형태
        nxG = nx.DiGraph(sortup(self.edges))
        paths = []
        while nxG.edges:
            path = nx.dag_longest_path(nxG)
            paths.append(path)
            for x, y in zip(path, path[1:]):
                nxG.remove_edge(x, y)
        nxG = nx.Graph([(x, y) for x, y in self.confounded_dict.values()])
        bipaths = []
        while nxG.edges:
            temppaths = []
            for x, y in itertools.combinations(sortup(nxG.nodes), 2):
                for spath in nx.all_simple_paths(nxG, x, y):
                    temppaths.append(spath)
            selected = sorted(temppaths, key=lambda _spath: len(_spath), reverse=True)[0]
            bipaths.append(selected)
            for x, y in zip(selected, selected[1:]):
                nxG.remove_edge(x, y)

        modified = True
        while modified:
            modified = False
            for i, path1 in enumerate(bipaths):
                for j, path2 in enumerate(bipaths[i + 1:], i + 1):
                    if path1[-1] == path2[0]:
                        newpath = path1 + path2[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                    elif path1[0] == path2[-1]:
                        newpath = path2 + path1[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                    elif path1[0] == path2[0]:
                        newpath = list(reversed(path2)) + path1[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                    elif path1[-1] == path2[-1]:
                        newpath = path2 + list(reversed(path1))[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                modified = path1 != bipaths[i]
                if modified:
                    break

        # a -> b -> c
        # e -> d -> c
        # == a->b->c<-d<-e
        paths_string = [' ⟶ '.join(path) for path in paths]
        bipaths_string = [' ⟷ '.join(path) for path in bipaths]
        alone = self.V - {x for path in paths for x in path} - {x for path in bipaths for x in path}
        if alone:
            return f'[{",".join([str(x) for x in alone])} / ' + (', '.join(paths_string) + ' / ' + ', '.join(bipaths_string)) + ']'
        else:
            return f'[' + (', '.join(paths_string) + ' / ' + ', '.join(bipaths_string)) + ']'
    
    def draw_graph(self):   # HTML로 그래프 그려줌 -> interactive하게 바꾸기 

        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        G.add_nodes_from(self.V)
        
        # Add directed edges
        G.add_edges_from(self.edges)
        
        # Create a separate graph for bidirected edges to handle them differently in drawing
        G_bi = nx.MultiDiGraph()
        G_bi.add_nodes_from(self.V)
        G_bi.add_edges_from([(u, v) for u, v, _ in self.bidirected_edges])
        G_bi.add_edges_from([(v, u) for u, v, _ in self.bidirected_edges])  # Add reverse direction for bidirected
        
        # Draw the directed graph
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', connectionstyle='arc3,rad=0.1')
        
        # Draw bidirected edges with a different style
        nx.draw_networkx_edges(G_bi, pos, edge_color='red', connectionstyle='arc')

        plt.show()


class StructuralCausalModel:
    def __init__(self, G: CausalDiagram, F=None, P_U=None, D=None, more_U=None):
        self.G = G                                              # Causal diagram
        self.F = F                                              # Fuctional relationship
        self.P_U = P_U                                          # Distribution of U : UC와 Uv를 모두 입력
        self.D = with_default(D, defaultdict(lambda: (0, 1)))   # 변수들의 domain : 기본으로 binary ( 0 or 1 )
        self.more_U = set() if more_U is None else set(more_U)  # G에 없는 U : UC말고 Uv 입력

        self.query00 = functools.lru_cache(1024)(self.query00)

    # V, U, P_U, F를 이용해서 intervention, condition을 고려한 data generation + 보고싶은 outcome 변수의 확률 계산
    def query(self, outcome: Tuple, condition: dict = None, intervention: dict = None, verbose=False) -> defaultdict:
        """
        outcome : ('Y')
        condition : {'Z': 0}
        intervention : {'X': 1}
        """
        if condition is None:
            condition = dict()
        if intervention is None:
            intervention = dict()
        new_condition = tuple(sorted([(x, y) for x, y in condition.items()]))
        new_intervention = tuple(sorted([(x, y) for x, y in intervention.items()]))
        return self.query00(outcome, new_condition, new_intervention, verbose)

    def query00(self, outcome: Tuple, condition: Tuple, intervention: Tuple, verbose=False) -> defaultdict:
        condition = dict(condition)
        intervention = dict(intervention)

        prob_outcome = defaultdict(lambda: 0)

        U = list(sorted(self.G.U | self.more_U))    # UC와 Uv를 합쳐서 sort
        D = self.D                                  # binary domain
        P_U = self.P_U                              # U들의 확률분포
        V_ordered = self.G.causal_order()           # V의 topological order
        if verbose:                                 
            print(f"ORDER: {V_ordered}")
        normalizer = 0                              
        
        # print([D[U_i] for U_i in U])    # [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        # print(*[D[U_i] for U_i in U])   # (0, 1) (0, 1) (0, 1) (0, 1) (0, 1) (0, 1) (0, 1) (0, 1)
        for u in product(*[D[U_i] for U_i in U]):  # d^|U|  # U들의 모든 combinations
            assigned = dict(zip(U, u))              # U와 realization을 dict로 묶음
            p_u = P_U(assigned)                     # 이런 U의 조합이 나올 확률
            if p_u == 0:                            
                continue
            # evaluate values                       
            for V_i in V_ordered:                                   # topological 순서로 V를 돌면서 assigned에 realization 추가(data generation)
                if V_i in intervention:                             # intervention된 경우 assigned를 intervention으로 고정
                    assigned[V_i] = intervention[V_i]               
                else:                                               # 아니면 F에 정의된 function에 의해 계산된 V를 추가
                    assigned[V_i] = self.F[V_i](assigned)  # pa_i including unobserved

            if not all(assigned[V_i] == condition[V_i] for V_i in condition):   # condition과 realization이 일치하지 않으면 넘어감
                continue
            normalizer += p_u                                                   # U 조합이 나올 확률을 계속 + -> condition 있는 경우 나눠주는 용도
            prob_outcome[tuple(assigned[V_i] for V_i in outcome)] += p_u        # outcome realization에 대한 확률+

            # print(prob_outcome, p_u) # outcome = ('Y') 면 0, 1일 확률, ('Y', 'Z')면 (0,0),(0,1),(1,0),(1,1)일 확률

        if prob_outcome:
            # normalize by prob condition
            return defaultdict(lambda: 0, {k: v / normalizer for k, v in prob_outcome.items()}) # normalizer로 나눈 확률 return
        else:
            return defaultdict(lambda: np.nan)  # nan or 0?


def quick_causal_diagram(paths, bidirectedpaths=None):
    if bidirectedpaths is None:
        bidirectedpaths = []
    dir_edges = []
    for path in paths:
        for x, y in zip(path, path[1:]):
            dir_edges.append((x, y))
    bidir_edges = []
    u_count = 0
    for path in bidirectedpaths:
        for x, y in zip(path, path[1:]):
            bidir_edges.append((x, y, 'U' + str(u_count)))
            u_count += 1
    return CausalDiagram(set(), dir_edges, bidir_edges)

# __add__에 사용되는 2개의 diagram을 합치는 함수 : 노드, edge, UC 합침
def merge_two_cds(g1: CausalDiagram, g2: CausalDiagram) -> CausalDiagram:
    VV = g1.V | g2.V
    EE = set(g1.edges) | set(g2.edges)
    VWU = set(g1.confounded_to_3tuples()) | set(g2.confounded_to_3tuples())
    return CausalDiagram(VV, EE, VWU)


# CausalDiagram class에서 __repr__ 정의에 쓰임
def cd2qcd(G: CausalDiagram) -> str:
    nxG = nx.DiGraph(sortup(G.edges))
    paths = []
    while nxG.edges:
        path = nx.dag_longest_path(nxG)
        paths.append(path)
        for x, y in zip(path, path[1:]):
            nxG.remove_edge(x, y)
    nxG = nx.Graph([(x, y) for x, y in G.confounded_dict.values()])
    bipaths = []
    while nxG.edges:
        temppaths = []
        for x, y in itertools.combinations(sortup(nxG.nodes), 2):
            for spath in nx.all_simple_paths(nxG, x, y):
                temppaths.append(spath)
        selected = sorted(temppaths, key=lambda _spath: len(_spath), reverse=True)[0]
        bipaths.append(selected)
        for x, y in zip(selected, selected[1:]):
            nxG.remove_edge(x, y)

    if all(len(v) == 1 for path in paths for v in path) and all(len(v) == 1 for path in bipaths for v in path):
        paths = [''.join(path) for path in paths]
        bipaths = [''.join(path) for path in bipaths]

    return f'qcd({paths}, {bipaths})'
