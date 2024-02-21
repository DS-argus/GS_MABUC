from npsem.model import CD
import matplotlib.pyplot as plt
import causaleffect

if __name__ == "__main__":

    G = causaleffect.createGraph(['W2->X', 'X->Y', 'W1->W2', 'W3->W4', 'W4->X',
                                  'W1<->X', 'W2<->W3', 'W3<->W5', 'W5<->W4', 'W1<->W3', 'W1<->Y'])
    P = causaleffect.ID({'Y'}, {'X'}, G)
    print(P.printLatex())