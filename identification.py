from model import CausalDiagram
from probability import Probability, get_new_probability

# Define exceptions that can occur.
class HedgeFound(Exception):
    '''Exception raised when a hedge is found.'''

    def __init__(self, g1, g2, message="Causal effect not identifiable. A hedge has been found:"):
        self._message = message
        super().__init__(self._message + f"\n\nC-Forest 1:\n: {g1} \n\nC-Forest 2:\n: {g2}")


def ID(y: set, x: set, P: "Probability", G: "CausalDiagram", verbose: bool = False, tab: int = 0):
    """
    OUTPUT : Expression in Latex 

    Shpitser, Pearl 2006
    [Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models]
    """
    # breakpoint()
    # Line 1
    if len(x)==0:
        if verbose: print('line 1')
        P_out = P.copy()
        if P_out._recursive:
            P_out._sumset = P._sumset.union(G.V.difference(y))
        else:
            P_out._var = y
        return P_out
        
    # line 2
    anc = G.An(y)
    if len(G.V- anc)>0:
        if verbose: print('line 2')
        
        P_out = P.copy()
        if P_out._recursive:
            P_out._sumset = P._sumset.union(G.V.difference(anc))
        else:
            P_out._var = anc
        return ID(y, x.intersection(anc), P_out, G.induced(anc), tab=tab + 1)
    
    # line 3
    # breakpoint()
    G_do_x = G.do(x)
    anc_x = G_do_x.An(y)
    W = G.V-x-anc_x
    if len(W)>0:
        if verbose: print('line 3')
        return ID(y, x.union(W), P, G, tab=tab + 1)

    # line 4
    CCs = G[G.V-x].c_components
    if len(CCs) > 1:
        if verbose: print('line 4')
        probabilities = set()
        for CC in CCs:
            probabilities.add(
                ID(CC, G.V.difference(CC), P, G, tab=tab + 1))
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
            order = G.causal_order()
            # if len(S) == 1:
            #     (vertex,) = S
            #     cond = get_previous_order(vertex, V, ordering)
            #     P_out = get_new_probability(P, {vertex}, cond)
            #     P_out._sumset = P_out._sumset.union(S.difference(Y))
            #     return P_out

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
                # if len(cc) == 1:
                #     (vertex,) = cc
                #     order = G.causal_order()
                #     prev = set(order[:order.index(vertex)])
                #     cond = (prev & cc) | (prev - cc)
                #     P_out = get_new_probability(P, {vertex}, cond)

                #     return ID(y, x.intersection(cc), Probability(recursive=True, children=probabilities),
                #             G.induced(cc), ordering, tab=tab + 1)

                probabilities = set()
                for vertex in S_prime:
                    # breakpoint()
                    order = G.causal_order()
                    prev = set(order[:order.index(vertex)])
                    cond = (prev & cc) | (prev - cc)
                    P_out = get_new_probability(P, {vertex}, cond)
                    probabilities.add(P_out)
                return ID(y, x.intersection(S_prime), Probability(recursive=True, children=probabilities),
                            G.induced(S_prime), tab=tab + 1)


def conditional_ID(do, condition):
    """
    Shpitser, Pearl 2006
    [Identification of Conditional Interventional Distributions]
    """

    pass

if __name__ == "__main__":

    # # Simple cases
    # G = CausalDiagram({'X', 'Z', 'Y'}, 
    #                   [('X', 'Z'), ('Z', 'Y')],
    #                   [])
    
    # P1 = ID(y=set("Y"), x=set("X"), P=Probability(var=G.V), G=G)
    # print(P1.printLatex())
    
    # P2 = ID(y=set("Y"), x=set("Z"), P=Probability(var=G.V), G=G)
    # print(P2.printLatex())

    # P3 = ID(y=set("Y"), x=set(), P=Probability(var=G.V), G=G)
    # print(P3.printLatex())


    # # Figure 1.(a) in Paper
    # G = CausalDiagram({'W1', 'W2', 'X', 'Y1', 'Y2'}, 
    #                   [('W1', 'X'), ('X', 'Y1'), ('W2', 'Y2')],
    #                   [('W1', 'W2', 'U_W1W2'), ('W1', 'Y2', 'U_W1Y2'), ('W2', 'X', 'U_W2X'), ('W1', 'Y1', 'U_W1Y1')])

    # P4 = ID(y=set(["Y1", "Y2"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    # print(P4.printLatex())

    # # Figure 1.(b) in Paper
    # G = CausalDiagram({'W1', 'W2', 'X', 'Y1', 'Y2'}, 
    #                   [('W1', 'X'), ('W1', 'W2'), ('X', 'Y1'), ('W2', 'Y2')],
    #                   [('W1', 'W2', 'U_W1W2'), ('W1', 'Y2', 'U_W1Y2'), ('W2', 'X', 'U_W2X'), ('W1', 'Y1', 'U_W1Y1')])

    # P5 = ID(y=set(["Y1", "Y2"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    # print(P5.printLatex())

    # # Figure 2.(a) in Paper
    # G = CausalDiagram({'X', 'Y'},
    #                   [('X', 'Y')],
    #                   [('X', 'Y', 'U_XY')])
    
    # P6 = ID(y=set(["Y"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    # print(P6.printLatex())

    # # Figure 2.(b) in Paper
    # G = CausalDiagram({'X', 'Z', 'Y'},
    #                   [('X', 'Z'), ('Z', 'Y')],
    #                   [('X', 'Z', 'U_XZ')])
    
    # P6 = ID(y=set(["Y"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    # print(P6.printLatex())


    # # Figure 2.(c) in Paper   
    # G = CausalDiagram({'X', 'Z', 'Y'},
    #                   [('X', 'Z'), ('Z', 'Y'), ('X', 'Y')],
    #                   [('X', 'Z', 'U_XZ')])
    
    # P7 = ID(y=set(["Y"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    # print(P7.printLatex())

    # # Figure 2.(d) in Paper   
    # G = CausalDiagram({'X', 'Z', 'Y'},
    #                   [('Z', 'Y'), ('X', 'Y')],
    #                   [('X', 'Z', 'U_XZ'), ('Z', 'Y', 'U_ZY')])
    
    # P8 = ID(y=set(["Y"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    # print(P8.printLatex())

    # # Figure 2.(e) in Paper   
    # G = CausalDiagram({'X', 'Z', 'Y'},
    #                   [('Z', 'X'), ('X', 'Y')],
    #                   [('X', 'Z', 'U_XZ'), ('Z', 'Y', 'U_ZY')])
    
    # P9 = ID(y=set(["Y"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    # print(P9.printLatex())

    # # Figure 2.(f) in Paper   
    # G = CausalDiagram({'X', 'Z', 'Y'},
    #                   [('X', 'Z'), ('Z', 'Y')],
    #                   [('X', 'Y', 'U_XY'), ('Z', 'Y', 'U_ZY')])
    
    # P10 = ID(y=set(["Y"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    # print(P10.printLatex())

    # # Figure 2.(g) in Paper   
    # G = CausalDiagram({'X', 'Z1', 'Z2', 'Y'},
    #                   [('X', 'Z1'), ('Z1', 'Y'), ('Z2', 'Y')],
    #                   [('X', 'Z2', 'U_XZ2'), ('Z1', 'Z2', 'U_Z1Z2')])
    
    # P11 = ID(y=set(["Y"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    # print(P11.printLatex())

    # Figure 2.(h) in Paper   
    G = CausalDiagram({'X', 'Z', 'W', 'Y'},
                      [('Z', 'X'), ('X', 'W'), ('W', 'Y')],
                      [('Z', 'X', 'U_ZX'), ('Z', 'W', 'U_ZW'), ('Z', 'Y', 'U_ZY'), ('X', 'Y', 'U_XY')])
    
    P12 = ID(y=set(["Y"]), x=set(["X"]), P=Probability(var=G.V), G=G)
    print(P12.printLatex())