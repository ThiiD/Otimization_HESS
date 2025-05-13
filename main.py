import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize

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

Cap_b = 40.0            # Capacidade da bateria em Ah
T_xb = 6                # Multiplicador da capacidade da bateria
Cap_uc = 200.0          # Capacidade do supercapacitor em Ah
T_xuc = 6               # Multiplicador da capacidade do supercapacitor


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
#   I_bmax: corrente máxima da bateria (em A)
#   I_ucmax: corrente máxima do supercapacitor (em A)
#   E_rej: energia rejeitada pelo sistema de armazenamento de energia (em kWh)
#   DoD_bat: Variação do Depth of Discharge da bateria
#   DoD_uc: Variação do Depth of Discharge do supercapacitor

# Equações do problema:
#   min Pt = (Np_b * Nm_b * Ns_b) * Pb + (Np_uc * Nm_uc * Ns_uc) * Puc
#   min Vt = (Np_b * Nm_b * Ns_b) * Vb + (Np_uc * Nm_uc * Ns_uc) * Vuc
#   min Wt = (Np_b * Nm_b * Ns_b) * Wb + (Np_uc * Nm_uc * Ns_uc) * Wuc
#   max It = (Np_b * Cap_b * T_xb) + (Np_uc * Cap_uc * T_xuc)
#   s.a

        
#   E_rej = (Np_b * Nm_b * Ns_b) * Pb * (1 - DoD_bat) + (Np_uc * Nm_uc * Ns_uc) * Puc * (1 - DoD_uc)
#   DoD_bat = (Np_b * Nm_b * Ns_b) * Pb / ((Np_b * Nm_b * Ns_b) * Pb + (Np_uc * Nm_uc * Ns_uc) * Puc)


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,  # Np_b e Np_uc
                         n_obj=1,   # Função objetivo combinada
                         n_ieq_constr=0,  # Sem restrições
                         xl=np.array([1, 1]),  # Mínimo de 1 para cada variável
                         xu=np.array([10, 10]),  # Máximo de 10 para cada variável
                         type_var=np.int64)  # Especificando que as variáveis são inteiras
        
        # Pesos para cada função objetivo
        self.weights = np.array([0.01, 0.01, 0.01, 0.97])  # Soma deve ser 1.0 (Custo, Volume, Peso, Corrente)
        
        # Calculando valores de referência para normalização usando xl e xu
        # Custo máximo possível
        self.custo_ref = (self.xu[0] * Nm_b * Ns_b * Pb + 
                         self.xu[1] * Nm_uc * Ns_uc * Puc)
        
        # Volume máximo possível
        self.volume_ref = (self.xu[0] * Nm_b * Ns_b * Vb + 
                          self.xu[1] * Nm_uc * Ns_uc * Vuc)
        
        # Peso máximo possível
        self.peso_ref = (self.xu[0] * Nm_b * Ns_b * Wb + 
                        self.xu[1] * Nm_uc * Ns_uc * Wuc)
        
        # Corrente máxima possível
        self.corrente_ref = (self.xu[0] * Cap_b * T_xb + 
                            self.xu[1] * Cap_uc * T_xuc)

    def _evaluate(self, x, out, *args, **kwargs):
        # Variáveis de decisão (garantindo valores inteiros)
        Np_b = int(round(x[0]))   # Número de baterias em paralelo
        Np_uc = int(round(x[1]))  # Número de supercapacitores em paralelo

        # Cálculo do número total de componentes
        total_baterias = Np_b * Nm_b * Ns_b
        total_supercaps = Np_uc * Nm_uc * Ns_uc

        # Cálculo do custo total (em dólares)
        f1 = (total_baterias * Pb + total_supercaps * Puc) / self.custo_ref * self.weights[0]

        # Cálculo do volume total (em L)
        f2 = (total_baterias * Vb + total_supercaps * Vuc) / self.volume_ref * self.weights[1]

        # Cálculo do peso total (em kg)
        f3 = (total_baterias * Wb + total_supercaps * Wuc) / self.peso_ref * self.weights[2]

        # Cálculo da corrente máxima total (em A)
        I_bmax = Np_b * Cap_b * T_xb
        I_ucmax = Np_uc * Cap_uc * T_xuc
        f4 = -(I_bmax + I_ucmax) / self.corrente_ref * self.weights[3]

        # Função objetivo combinada
        out["F"] = [f1 + f2 + f3 + f4]  # Objetivo único combinado
        out["G"] = []  # Sem restrições

        # Print de debug para a primeira avaliação
        if not hasattr(self, 'debug_printed'):
            print("\nValores das constantes:")
            print(f"Pb: {Pb}, Puc: {Puc}")
            print(f"Vb: {Vb}, Vuc: {Vuc}")
            print(f"Wb: {Wb}, Wuc: {Wuc}")
            print(f"Nm_b: {Nm_b}, Ns_b: {Ns_b}")
            print(f"Nm_uc: {Nm_uc}, Ns_uc: {Ns_uc}")
            print(f"Cap_b: {Cap_b}, T_xb: {T_xb}")
            print(f"Cap_uc: {Cap_uc}, T_xuc: {T_xuc}")
            print("\nExemplo de cálculo para Np_b=5.0, Np_uc=5.0:")
            print(f"Total baterias: {5.0 * Nm_b * Ns_b}")
            print(f"Total supercaps: {5.0 * Nm_uc * Ns_uc}")
            print(f"Custo: {5.0 * Nm_b * Ns_b * Pb + 5.0 * Nm_uc * Ns_uc * Puc}")
            print(f"Volume: {5.0 * Nm_b * Ns_b * Vb + 5.0 * Nm_uc * Ns_uc * Vuc}")
            print(f"Peso: {5.0 * Nm_b * Ns_b * Wb + 5.0 * Nm_uc * Ns_uc * Wuc}")
            print(f"Corrente máxima bateria: {5.0 * Cap_b * T_xb}")
            print(f"Corrente máxima supercap: {5.0 * Cap_uc * T_xuc}")
            print(f"Corrente máxima total: {5.0 * Cap_b * T_xb + 5.0 * Cap_uc * T_xuc}")
            self.debug_printed = True

problem = MyProblem()

algorithm = GA(
    pop_size=200,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=15, prob=0.2),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 200)

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

# Visualização dos resultados
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')

# Calculando os valores originais para a solução otimizada
Np_b = int(round(X[0]))
Np_uc = int(round(X[1]))
total_baterias = Np_b * Nm_b * Ns_b
total_supercaps = Np_uc * Nm_uc * Ns_uc

custo = total_baterias * Pb + total_supercaps * Puc
volume = total_baterias * Vb + total_supercaps * Vuc
peso = total_baterias * Wb + total_supercaps * Wuc
corrente = Np_b * Cap_b * T_xb + Np_uc * Cap_uc * T_xuc

ax.scatter(custo, volume, peso, c='red', marker='o', s=100)
ax.set_xlabel('Custo Total ($)')
ax.set_ylabel('Volume Total (L)')
ax.set_zlabel('Peso Total (kg)')
ax.set_title('Solução Otimizada')
plt.show(block=False)

# Visualização do espaço de design
plt.figure(figsize=(7, 5))
plt.scatter(X[0], X[1], s=100, facecolors='r', edgecolors='r', zorder=3)
plt.xlabel('Número de Baterias em Paralelo')
plt.ylabel('Número de Supercapacitores em Paralelo')
plt.title("Espaço de Design")
plt.grid(zorder=1)
plt.xticks(np.arange(1, 11, 1))
plt.yticks(np.arange(1, 11, 1))

# Imprimir resultados
print("\nResultados da Otimização:")
print("------------------------")
print("\nSolução Otimizada:")
print(f"Número de baterias em paralelo: {Np_b}")
print(f"Número de supercapacitores em paralelo: {Np_uc}")
print(f"Custo Total: ${int(round(custo))}")
print(f"Volume Total: {int(round(volume))} L")
print(f"Peso Total: {int(round(peso))} kg")
print(f"Corrente Máxima Total: {int(round(corrente))} A")

plt.show(block=True)