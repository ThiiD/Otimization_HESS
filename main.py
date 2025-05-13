import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2

import numpy as np
from pymoo.core.problem import ElementwiseProblem

# Definição dos parametros do problema
Pb = 28.00              # Preço da bateria em dolares (Fonte: data_sources.xlsx)
Puc= 53.75              # Preço do supercapacitor em dolares (Fonte: data_sources.xlsx)

Vb = 0.596              # Volume da bateria em L (Fonte: data_sources.xlsx)
Vuc = 0.00837           # Volume do supercapacitor em L (Fonte: data_sources.xlsx)

Wb = 1.060              # Peso da bateria em kg (Fonte: data_sources.xlsx)
Wuc = 0.460             # Peso do supercapacitor em kg (Fonte: data_sources.xlsx)

Ns_b = 16               # Número de baterias em serie
Nm_b = 24               # Número de módulos de baterias
Ns_uc = 16              # Número de supercapacitores em serie
Nm_uc = 20              # Número de módulos de supercapacitores


# Definição do problema de otimização
# A principio, Np_b (número de baterias em paralelo) e Np_uc (número de supercapacitores em paralelo) são variáveis de decisão
# A principio, o problema é definido como um problema de otimização multiobjetivo, em que o objetivo é minimizar o custo total, o volume total e o peso total do sistema de armazenamento de energia
# Posteriormente, irá entrar como variavel de decisão o threshold de potência, que influenciará na energia rejeitada.
# A variação de DoD também entrará como valores de saída do problema de otimização

# Entradas do problema:
#   Np_b: número de baterias em paralelo
#   Np_uc: número de supercapacitores em paralelo
#   Pth: threshold de potência (em W) - valor de decisão

# Saídas do problema:
#   Custo_total: custo total do sistema de armazenamento de energia (em dolares)
#   Volume_total: volume total do sistema de armazenamento de energia (em L)
#   Peso_total: peso total do sistema de armazenamento de energia (em kg)
#   E_rej: energia rejeitada pelo sistema de armazenamento de energia (em kWh)
#   DoD_bat: Variação do Depth of Discharge da bateria
#   DoD_uc: Variação do Depth of Discharge do supercapacitor

# Equações do problema:
#   min Pt = (Np_b * Nm_b * Ns_b) * Pb + (Np_uc * Nm_uc * Ns_uc) * Puc
#   min Vt = (Np_b * Nm_b * Ns_b) * Vb + (Np_uc * Nm_uc * Ns_uc) * Vuc
#   min Wt = (Np_b * Nm_b * Ns_b) * Wb + (Np_uc * Nm_uc * Ns_uc) * Wuc
#   s.a

        
#   E_rej = (Np_b * Nm_b * Ns_b) * Pb * (1 - DoD_bat) + (Np_uc * Nm_uc * Ns_uc) * Puc * (1 - DoD_uc)
#   DoD_bat = (Np_b * Nm_b * Ns_b) * Pb / ((Np_b * Nm_b * Ns_b) * Pb + (Np_uc * Nm_uc * Ns_uc) * Puc)


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


problem = MyProblem()

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

from pymoo.termination import get_termination

termination = get_termination("n_gen", 40)

from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

X = res.X
F = res.F

print(f'X: {X}')
print(f'F: {F}')

import matplotlib.pyplot as plt
xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
plt.xlim(xl[0], xu[0])
plt.ylim(xl[1], xu[1])
plt.title("Design Space")
plt.show(block=False)

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()