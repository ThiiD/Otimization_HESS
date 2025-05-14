import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.core.problem import ElementwiseProblem

from simulation import Simulation

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
Cap_uc = 280.0          # Capacidade do supercapacitor em Ah
T_xuc = 3               # Multiplicador da capacidade do supercapacitor


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
                         n_obj=5,   # Custo total, Volume total, Peso total, Corrente máxima e Energia rejeitada
                         n_ieq_constr=0,  # Sem restrições
                         xl=np.array([1, 1]),  # Mínimo de 1 para cada variável
                         xu=np.array([10, 10]),  # Máximo de 10 para cada variável
                         type_var=np.int64)  # Especificando que as variáveis são inteiros
        
        # Pesos para cada função objetivo
        self.weights = np.array([0.1, 0.25, 0.01, 0.25, 0.3])  # Soma deve ser 1.0 (Custo, Volume, Peso, Corrente, Energia rejeitada)
        
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
        
        # Dicionário para armazenar resultados das simulações
        self.simulation_cache = {}

        # Calculando energia rejeitada de referência usando valores mínimos (pior caso)
        print("Calculando energia rejeitada de referência (pior caso)...")
        sim = Simulation()
        sim.setParam_Batt(C=Cap_b, Ns=Ns_b, Np=self.xl[0], Nm=Nm_b, Vnom=3.2, SoC=50)
        sim.setParam_UC(C=3400, Ns=Ns_uc, Np=self.xl[1], Nm=Nm_uc, Vnom=3, SoC=50)
        
        # Carregar dados do arquivo Excel   
        data = r"data\CR-3112_28-09-24_AGGREGATED.xlsx"
        sheet = "Dados"
        
        # Simular o sistema com o threshold especificado
        sim.simulate(data, sheet, threshold=1000)
        
        # Armazenar a energia rejeitada total como referência
        self.energy_reject_ref = sum(sim._p_reject) / 3600  # Convertendo de W para Wh
        print(f"Energia rejeitada de referência (pior caso): {self.energy_reject_ref:.2f} Wh")

    def _evaluate(self, x, out, *args, **kwargs):
        # Variáveis de decisão (garantindo valores inteiros)
        Np_b = int(round(x[0]))   # Número de baterias em paralelo
        Np_uc = int(round(x[1]))  # Número de supercapacitores em paralelo

        # Cálculo do número total de componentes
        total_baterias = Np_b * Nm_b * Ns_b
        total_supercaps = Np_uc * Nm_uc * Ns_uc

        # Cálculo do custo total (em dólares) - peso 0.1
        f1 = (total_baterias * Pb + total_supercaps * Puc) / self.custo_ref * self.weights[0]

        # Cálculo do volume total (em L) - peso 0.25
        f2 = (total_baterias * Vb + total_supercaps * Vuc) / self.volume_ref * self.weights[1]

        # Cálculo do peso total (em kg) - peso 0.01
        f3 = (total_baterias * Wb + total_supercaps * Wuc) / self.peso_ref * self.weights[2]

        # Cálculo da corrente máxima total (em A) - peso 0.25
        I_bmax = Np_b * Cap_b * T_xb
        I_ucmax = Np_uc * Cap_uc * T_xuc
        f4 = -(I_bmax + I_ucmax) / self.corrente_ref * self.weights[3]

        # Verifica se já temos o resultado da simulação no cache
        cache_key = (Np_b, Np_uc)
        if cache_key not in self.simulation_cache:
            # Simulação para calcular energia rejeitada
            sim = Simulation()
            sim.setParam_Batt(C=Cap_b, Ns=Ns_b, Np=Np_b, Nm=Nm_b, Vnom=3.2, SoC=50)
            sim.setParam_UC(C=3400, Ns=Ns_uc, Np=Np_uc, Nm=Nm_uc, Vnom=3, SoC=50)
            
            # Carregar dados do arquivo Excel   
            data = r"data\CR-3112_28-09-24_AGGREGATED.xlsx"
            sheet = "Dados"
            
            # Simular o sistema com o threshold especificado
            sim.simulate(data, sheet, threshold=1000)
            
            # Armazena o resultado no cache (convertendo para Wh)
            self.simulation_cache[cache_key] = sum(sim._p_reject) / 3600
        
        # Usa o resultado do cache
        energy_reject = self.simulation_cache[cache_key] / self.energy_reject_ref * self.weights[4]
        f5 = energy_reject

        out["F"] = [f1, f2, f3, f4, f5]  # Objetivos normalizados e ponderados
        out["G"] = []  # Sem restrições


problem = MyProblem()

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

algorithm = NSGA2(
    pop_size=50,
    n_offsprings=25,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=15, prob=0.2),
    eliminate_duplicates=True
)

from pymoo.termination import get_termination

termination = get_termination("n_gen", 20)

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

# Visualização dos resultados
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
if F is not None:
    # Convertendo os valores normalizados de volta para os valores originais
    F_original = F.copy()
    F_original[:, 0] *= problem.custo_ref
    F_original[:, 1] *= problem.volume_ref
    F_original[:, 2] *= problem.peso_ref
    F_original[:, 3] *= -problem.corrente_ref  # Invertendo o sinal negativo
    F_original[:, 4] *= problem.energy_reject_ref  # Convertendo energia rejeitada
    ax.scatter(F_original[:, 0], F_original[:, 1], F_original[:, 4], c='blue', marker='o')
ax.set_xlabel('Custo Total ($)')
ax.set_ylabel('Volume Total (L)')
ax.set_zlabel('Energia Rejeitada (Wh)')
ax.set_title('Espaço de Objetivos')
plt.show(block = False)

# Visualização do espaço de design
plt.figure(figsize=(7, 5))
# Convertendo os valores para inteiros antes de plotar
X_int = np.round(X).astype(int)
plt.scatter(X_int[:, 0], X_int[:, 1], s=30, facecolors='r', edgecolors='r', zorder = 3)
plt.xlabel('Número de Baterias em Paralelo')
plt.ylabel('Número de Supercapacitores em Paralelo')
plt.title("Espaço de Design")
# Adicionando grade para melhor visualização dos valores inteiros
plt.grid(zorder = 1)
# Definindo os ticks dos eixos para mostrar apenas valores inteiros
plt.xticks(np.arange(1, 11, 1))
plt.yticks(np.arange(1, 11, 1))
plt.xlim(problem.xl[0]-1, problem.xu[0]+1)
plt.ylim(problem.xl[1]-1, problem.xu[1]+1)


# Imprimir resultados
print("\nResultados da Otimização:")
print("------------------------")
for i in range(min(10, len(X))):  # Mostrar as 10 primeiras soluções
    print(f"\nSolução {i+1}:")
    print(f"Número de baterias em paralelo: {int(round(X[i,0]))}")
    print(f"Número de supercapacitores em paralelo: {int(round(X[i,1]))}")
    print(f"Custo Total: ${int(round(F[i,0] * problem.custo_ref))}")
    print(f"Volume Total: {int(round(F[i,1] * problem.volume_ref))} L")
    print(f"Peso Total: {int(round(F[i,2] * problem.peso_ref))} kg")
    print(f"Corrente Máxima Total: {int(round(-F[i,3] * problem.corrente_ref))} A")
    print(f"Energia Rejeitada Total: {int(round(F[i,4] * problem.energy_reject_ref))} Wh")

plt.show(block = True)