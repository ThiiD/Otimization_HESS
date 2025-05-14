import pandas as pd
import numpy as np

class Batt():
    def __init__(self):
        """Inicializa uma bateria com valores padrão"""
        self._C = 40
        self._Ns = 16
        self._Np = 3
        self._Nm = 24
        self._Vnom = 3.25
        self._SoC = 50
        self._min_SoC = 10                                                                          # SoC mínimo permitido (%)
        self._max_SoC = 90                                                                          # SoC máximo permitido (%)
        self._v_cel = self.LUT(self._SoC)                                                           # Calcula v_cel usando o SoC inicial
        self._v_banco = self._v_cel * self._Ns * self._Nm
        self._total_energy = (self._Np * self._C) * (self._Ns * self._Nm * self._Vnom)              # Wh
        self._SoC_Energy = (self._SoC/100) * self._total_energy

    def setParams(self, C: float, Ns: int, Np: int, Nm: int, Vnom: float, SoC: float) -> None:
        """
        Configura os parâmetros da bateria
        :param float C: Taxa de descarga da bateria (Ah)
        :param int Ns: Número de baterias em série
        :param int Np: Número de baterias em paralelo
        :param int Nm: Número de módulos
        :param float Vnom: Tensão nominal por célula (V)
        :param float SoC: Estado de carga inicial da bateria (%)
        :raises ValueError: Se os parâmetros forem inválidos
        """
        if any(x <= 0 for x in [C, Ns, Np, Nm, Vnom]):
            raise ValueError("Todos os parâmetros devem ser positivos")
        
        if not self._min_SoC <= SoC <= self._max_SoC:
            raise ValueError(f"SoC deve estar entre {self._min_SoC}% e {self._max_SoC}%")

        self._C = C
        self._Ns = Ns
        self._Np = Np
        self._Nm = Nm
        self._Vnom = Vnom
        self._SoC = SoC
        self._v_cel = self.LUT(self._SoC)  # Calcula v_cel usando o SoC inicial
        self._v_banco = self._v_cel * self._Ns * self._Nm
        self._total_energy = (Np * C) * (Ns * Nm * Vnom)  # Wh
        self._SoC_Energy = (SoC/100) * self._total_energy

    def LUT(self, SoC: float) -> float:
        """
        Calcula a tensão da bateria de acordo com o SoC usando interpolação
        :param float SoC: Estado de carga da célula (%)
        :return float: Tensão correspondente (V)
        """
        try:
            df = pd.read_csv("data\\LUT_batt.csv", sep=";")
            df["SoC"] = 100 - df["SoC"]  # Inverte SoC para corresponder ao padrão de carga
            indice_proximo = (df['SoC'] - SoC).abs().idxmin()
            tensao = df.loc[indice_proximo, 'Tensao']
            return float(tensao)
        except Exception as e:
            print(f"Erro ao ler LUT: {e}")
            return self._Vnom  # Retorna tensão nominal em caso de erro

    def Energy2SoC(self, energy: float) -> float:
        """
        Calcula o SoC da bateria baseado na energia armazenada
        :param float energy: Energia total armazenada (Wh)
        :return float: Estado de carga (%)
        """
        return (energy * 100) / self._total_energy

    def setCurrent(self, power: float) -> (float | float):
        """
        Calcula a corrente por célula baseada na potência requerida
        :param float power: Potência requerida (W)
        :return float i_sat: Corrente por célula (A)
        :return float p_reject: Potência rejeitada (kW)
        """
        i = power / self._v_banco
        i_sat = np.clip(i, -6 * self._Np * self._C, 6 * self._Np * self._C)  # Limita corrente em ambas direções
        i_reject = i - i_sat                                                 # Calcula corrente rejeitada
        p_reject = (i_reject * self._v_banco)/1000                           # Calcula potência rejeitada
        return i_sat, p_reject

    def getTotalEnergy(self) -> float:
        """
        Retorna a energia total armazenada na bateria
        :return float: Energia total (Wh)
        """
        return self._total_energy
    
    
    def updateEnergy(self, current: float, dt: float) -> (float | float | float):
        """
        Atualiza a energia total da bateria usando contador de Coulomb
        :param float current: Corrente da bateria (A, + carga, - descarga)
        :param float dt: Intervalo de tempo (s)
        :return float SoC: SoC da bateria
        :return float v_banco: Tensão total do pack de bateria
        :return float p_reject: Potência rejeitada (kW)
        """
        # Calcula variação de energia
        charge = -1 * current * dt / 3600  # Converte para horas
        energy_variation = self._v_banco * charge
        
        # Calcula nova energia
        new_energy = self._SoC_Energy + energy_variation
        
        # Limita energia entre mínimo e máximo
        clip_energy = np.clip(
            new_energy, 
            (self._min_SoC/100) * self._total_energy,
            (self._max_SoC/100) * self._total_energy
        )
        p_reject = ((new_energy - clip_energy) / dt) / 1000             # Potência rejeitada (kW)


        
        self._SoC_Energy = clip_energy
        self._SoC = self.Energy2SoC(clip_energy)
        self._v_banco = self._Ns * self._Nm * self.LUT(self._SoC)
        
        return self._SoC, self._v_banco, p_reject