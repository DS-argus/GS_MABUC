import copy
from typing import Iterable

class Probability:
    factors = []

    def __init__(self, vars: dict[frozenset: frozenset], sumset: set = None, num_sum: int = 0) -> None:
        
        if sumset:
            assert num_sum > 0

        self.vars = vars
        self.sumset = sumset        # summation할 변수
        self.num_sum = num_sum      # 지금부터 자기 포함 앞으로 몇개의 factor가 같은 sigma안에 묶이는지
        self.Pv = None

        if not Probability.factors:
            Probability.factors.append(self)


    def copy(self):
        return copy.deepcopy(self)
    

    def add_prob(self, prob: "Probability"):

        if isinstance(prob, Probability): 
            Probability.factors.append(prob)
        else:
            raise ValueError("Can only add instances of Probability")


    @property
    def characteristic(self):
        print(f"vars: {self.vars}")
        print(f"summation: {self.summation}")
        print(f"num_sum: {self.num_sum}")
    

    def estimands(self):
        return


    @staticmethod
    def printLatex():
        ans = ""
        summation_stack = []

        for factor in Probability.factors:
            if factor.sumset and factor.num_sum > 0:
                summation_stack.append({'sumset': factor.sumset, 'remaining': factor.num_sum})
                ans += "\\left("

            if factor.sumset:
                ans += f"\\sum_{{{', '.join(factor.sumset).lower()}}}"

            for var, conds in factor.vars.items():
                if conds:
                    prob_expression = "P({}|{})".format(", ".join(var).lower(), ", ".join(conds).lower())
                    prob_expression = f"P({', '.join(var).lower()}|{', '.join(conds).lower()})"
                else:
                    prob_expression = f"P({', '.join(var).lower()})"
                ans += prob_expression

                # Check if more than one var, but since we iterate over a dictionary, add this after the loop
                ans += " \\cdot "

            ans = ans.rstrip(" \\cdot ")  # Remove the last multiplication sign

            while summation_stack and summation_stack[-1]['remaining'] == 1:
                ans += "\\right)"
                summation_stack.pop()
            if summation_stack:
                summation_stack[-1]['remaining'] -= 1

            if factor != Probability.factors[-1] and not summation_stack:
                ans += " \\cdot "

        return ans


if __name__ == "__main__":

    P1 = Probability({frozenset(['Y']): frozenset(['X', 'Z'])})
    P2 = Probability({frozenset(['W', 'E']): frozenset(['K'])}, set(['W', 'E']), 3)
    P3 = Probability({frozenset(['Q' ,'O']): frozenset()}, set(['Q', 'X']), 2)
    P4 = Probability({frozenset(['A']): frozenset(['B', 'C', 'D', 'E'])})
    P5 = Probability({frozenset(['Q']): frozenset(['O'])})
    P6 = Probability({frozenset(['Q']): frozenset(['O'])})
    P7 = Probability({frozenset(['Q']): frozenset(['O'])})
    P1.add_prob(P2)
    P2.add_prob(P3)
    P3.add_prob(P4)
    P4.add_prob(P5)
    P5.add_prob(P6)
    P6.add_prob(P7)
    
    print(P1.printLatex())
    # print(P2.printLatex())
    # print(P3.printLatex())