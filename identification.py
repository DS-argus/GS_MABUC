from model import CausalDiagram
from probability import Probability, get_new_probability

# Define exceptions that can occur.
class HedgeFound(Exception):
    '''Exception raised when a hedge is found.'''

    def __init__(self, g1, g2, message="Causal effect not identifiable. A hedge has been found:"):
        self._message = message
        super().__init__(self._message + f"\n\nC-Forest 1:\n: {g1} \n\nC-Forest 2:\n: {g2}")

class ThicketFound(Exception):
    '''Exception raised when a thicket is found.'''

    def __init__(self, message="Causal effect not identifiable. A thicket has been found:"):
        self._message = message
        super().__init__(self._message)


def ID(y: set, x: set, P: "Probability", G: "CausalDiagram", verbose: bool = False, tab: int = 0):
    """
    OUTPUT : Expression in Latex 

    Shpitser, Pearl 2006
    [Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models]
    """
    order = G.causal_order()

    # breakpoint()
    # Line 1
    if len(x)==0:
        if verbose: print('line 1')
        P_out = P.copy()
        if P_out._recursive:
            P_out._sumset = P._sumset | (G.V - y)
        else:
            P_out._var = y
        
        return P_out
        
    # line 2
    anc = G.An(y)
    if len(G.V- anc)>0:
        if verbose: print('line 2')
        
        P_out = P.copy()
        if P_out._recursive:
            P_out._sumset = P._sumset | (G.V - anc)
        else:
            P_out._var = anc
        # print(P_out.printLatex())
        return ID(y, x.intersection(anc), P_out, G.induced(anc), verbose, tab=tab + 1)
    
    # line 3
    # breakpoint()
    G_do_x = G.do(x)
    anc_x = G_do_x.An(y)
    W = G.V-x-anc_x
    if len(W)>0:
        if verbose: print('line 3')
        # print(P.printLatex())
        return ID(y, x.union(W), P, G, verbose, tab=tab + 1)

    # line 4
    CCs = G[G.V-x].c_components
    if len(CCs) > 1:
        if verbose: print('line 4')
        probabilities = set()
        for CC in CCs:
            probabilities.add(
                ID(CC, G.V.difference(CC), P, G, verbose, tab=tab + 1))
        return Probability(recursive=True, children=probabilities, sumset=G.V.difference(y.union(x)))


    if len(CCs) == 1:
        for CC in CCs:
            S = CC

        # line 5
        if len(G.c_components) == 1:
            if verbose: print('line 5')
            raise HedgeFound(G, G.induced(S))
        
        # line 6
        if S in G.c_components:
            if verbose: print('line 6')
            probabilities = set()
            for vertex in S:
                cond = set(order[:order.index(vertex)])
                P_out = get_new_probability(P, {vertex}, cond)
                probabilities.add(P_out)
            return Probability(recursive=True, children=probabilities, sumset=S.difference(y))
        
        # line 7
        for cc in G.c_components:
            if S < cc:
                if verbose: print('line 7')
                S_prime = cc
                probabilities = set()
                for vertex in S_prime:
                    # breakpoint()
                    order = G.causal_order()
                    prev = set(order[:order.index(vertex)])
                    cond = (prev & cc) | (prev - cc)
                    P_out = get_new_probability(P, {vertex}, cond)
                    probabilities.add(P_out)
                
                print(Probability(recursive=True, children=probabilities).printLatex())
                
                return ID(y, x.intersection(S_prime), Probability(recursive=True, children=probabilities),
                            G.induced(S_prime), verbose, tab=tab + 1)


def gID(Y: set, X: set, Z:set, P: "Probability", G: "CausalDiagram", verbose: bool = False, tab: int = 0):
    # breakpoint()
    # line 2 -> 여기도 line 7 처럼 수정해야할듯 이건 맞는 것 같기도 하고....
    for z in Z:
        if X == z & G.V: # X = {'X2'}, Z = {{'X1', 'X2'}, {'X2', 'X3'}}, G.V = {'X2', 'Y'}라면 모든 z에 대해 성립
            if verbose: print('line 2')
            P_out = P.copy()
            P_out._do = (z-G.V) | X
            if P_out._recursive:
                P_out._sumset = P._sumset | (G.V - Y)
            else:
                P_out._var = Y
            
            return P_out

    # line 3
    anc = G.An(Y)
    if G.V != anc:
        if verbose: print('line 3')
        
        P_out = P.copy()
        if P_out._recursive:
            P_out._sumset = P._sumset | (G.V-anc)
        else:
            P_out._var = anc
        return gID(Y, X & anc, Z, P_out, G.induced(anc), verbose, tab=tab + 1)
    
    # line 4
    G_do_x = G.do(X)
    anc_x = G_do_x.An(Y)
    W = (G.V-X) - anc_x
    if len(W) > 0:
        if verbose: print("line 4")

        return gID(Y, X | W, Z, P, G, verbose, tab=tab+1)
    
    # line 6
    CCs = G[G.V-X].c_components
    # print(CCs)
    if len(CCs) > 1:
        if verbose: print("line 6")

        probabilities= set()
        for CC in CCs:
            probabilities.add(
                gID(CC, G.V-CC, Z, P, G, verbose, tab=tab+1))
        return Probability(recursive=True, children=probabilities, sumset=G.V - (Y|X))
        
    # line 7, 8
    for z in Z:
        if X >= z & G.V:
            if verbose: print("line 7")
            P_out = P.copy()
            P_out._do = (z-G.V) | (X&z)
            P_out._var = G.V
            result = subID(Y, X-z, P_out, G.induced(G.V-(z&X)), verbose, tab=tab+1) 
            if result:
                return result

    if verbose: print("line 8")
    raise ThicketFound()   
        

def subID(Y: set, X: set, Q: "Probability", G: "CausalDiagram", verbose: bool = False, tab: int = 0):

    order = G.causal_order()

    # 무조건 1개
    # breakpoint()
    S = next(iter(G.induced(G.V-X).c_components))
    # line 11
    if len(X)==0:
        if verbose: print('(subID) line 11')
        Q_out = Q.copy()
        if Q_out._recursive:
            Q_out._sumset = Q._sumset | (G.V - Y)
        else:
            Q_out._var = Y
        return Q_out

    # line 12
    anc = G.An(Y)
    if G.V != anc:
        if verbose: print('(subID) line 12')
        
        Q_out = Q.copy()
        if Q_out._recursive:
            Q_out._sumset = Q._sumset.union(G.V - anc)
        else:
            Q_out._var = anc
        return subID(Y, X & anc, Q_out, G.induced(anc), verbose, tab=tab + 1)
    
    # line 13
    CCs = G.c_components
    if len(CCs) == 1:
        if verbose: print("(subID) line 13")
        return None
    
    # line 14
    # breakpoint()
    if S in CCs:
        if verbose: print('(subID) line 14')
        probabilities = set()
        for vertex in Y:
            cond = set(order[:order.index(vertex)])
            P_out = get_new_probability(Q, {vertex}, cond=cond)
            probabilities.add(P_out)
        return Probability(recursive=True, children=probabilities, sumset=S-Y)
    
    # line 15
    for cc in CCs:
        if S < cc:
            if verbose: print('(subID) line 15')
            S_prime = cc
            probabilities = set()
            for vertex in S_prime:
                prev = set(order[:order.index(vertex)])
                cond = (prev & S_prime) | (prev - S_prime)
                P_out = get_new_probability(Q, {vertex}, cond=cond)
                probabilities.add(P_out)
            return subID(Y, X & S_prime, Probability(recursive=True, children=probabilities),
                        G.induced(S_prime), verbose, tab=tab + 1)

if __name__ == "__main__":

    # Figure 2

    # G = CausalDiagram({'X', 'Z1', 'Z2', 'Z3', 'Y'}, 
    #                     [('X', 'Z1'), ('Z1', 'Y'), ('Z2', 'X'), ('Z2', 'Z1'), ('Z2', 'Z3'), ('Z3', 'Y')],
    #                     [('X', 'Z2', 'U_XZ2'), ('Y', 'Z2', 'U_YZ2'), ('X', 'Z3', 'U_XZ3'), ('X', 'Y', 'U_XY')])

    # P = ID(y=set(['Y', 'Z1','Z2', 'Z3']), x=set(["X"]), P=Probability(var=G.V), G=G,verbose=True)
    # print(f"ID: {P.printLatex()}")

    G = CausalDiagram({'X', 'W1', 'W2', 'W3', 'W4', 'W5', 'Y'}, 
                    [('X', 'Y'), ('W1', 'W2'), ('W2', 'X'), ('W4', 'X'), ('W3', 'W4')],
                    [('W1', 'W3', 'U_W1W3'), ('W2', 'W3', 'U_W2W3'), ('W3', 'W5', 'U_W3W5'), ('W4', 'W5', 'U_W4W5'), ('W1', 'Y', 'U_W1Y'), ('W1', 'X', 'U_W1X')])

    P = ID(y=set(['Y']), x=set(["X"]), P=Probability(var=G.V), G=G,verbose=True)
    # P1 = gID(Y=set(['Y']), X=set(["X"]), Z=frozenset([frozenset()]), P=Probability(var=G.V), G=G)
    print(f"ID: {P.printLatex()}")