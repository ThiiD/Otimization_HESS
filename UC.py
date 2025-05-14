import numpy as np
from time import sleep
from math import sqrt

class Uc():
    def __init__(self):
        """Inicializa um banco de supercapacitores com valores padrão"""
        self._C = 3140                                                              
        self._Ns = 40                                                               
        self._Np = 10                                                               
        self._Nm = 5                                                                
        self._v_cap = 3                                                             
        
        self._SoC_max = 100                                                         
        self._SoC_min = 3                                                           

        # Calcula tensões do banco
        self._v_total = self._v_cap * self._Ns * self._Nm
        self._v_banco = self._v_total * (50/100)  # Inicia com 50% SoC
        
        # Calcula energia total e inicial
        self._C_eq = self._C * self._Np/(self._Ns * self._Nm)  # Capacitância equivalente
        self._total_energy = 0.5 * self._C_eq * (self._v_total**2)
        self._stored_energy = 0.5 * self._C_eq * (self._v_banco**2)
        
        self._SoC = 50  # Estado de carga inicial (%)

    def setParams(self, C: float, Ns: int, Np: int, Nm : int, Vnom: float, SoC: float) -> None:
        """Configura os parâmetros do banco de supercapacitores
        :param float C: Capacitância por célula (F)
        :param int Ns: Número de capacitores em série
        :param int Np: Número de strings em paralelo
        :param int Nm: Número de módulos em série
        :param float Vnom: Tensão nominal por célula (V)
        :param float SoC: Estado de carga inicial do banco (%)
        :raises ValueError: Se os parâmetros forem inválidos
        """
        if any(x <= 0 for x in [C, Ns, Np, Nm, Vnom]):
            raise ValueError("Todos os parâmetros devem ser positivos")
        
        if not 0 <= SoC <= 100:
            raise ValueError("SoC deve estar entre 0% e 100%")

        self._C = C
        self._Ns = Ns
        self._Np = Np
        self._Nm = Nm
        self._v_cap = Vnom
        self._SoC = SoC

        # Recalcula parâmetros
        self._v_total = self._v_cap * self._Ns * self._Nm
        self._C_eq = C * Np/(Ns * Nm)
        self._total_energy = 0.5 * self._C_eq * (self._v_total**2)

        
        self._stored_energy = self._total_energy*(self._SoC/100)

        self._v_banco = sqrt((2 * self._stored_energy) / self._C_eq)
        
        
        

    def energy2soc(self, energy: float) -> float:
        """
        Calcula SoC baseado na energia armazenada no banco de UC.
        :param float energy: Energia armazenada (J)
        :return float: Estado de carga do banco (%)
        """
        return (energy / self._total_energy) * 100

    def soc2energy(self, SoC: float) -> float:
        """
        Calcula energia armazenada no banco de UC baseada no SoC.
        :param float SoC: Estado de carga do banco (%)
        :return float: Energia armazenada (J)
        """
        return sqrt((2 * self._total_energy * (SoC / 100)) / self._C_eq)

    def voltage2energy(self, voltage: float) -> float:
        """
        Calcula energia armazenada no banco de UC baseada na tensão.
        :param float voltage: Tensão do banco (V)
        :return float: Energia armazenada (J)
        """
        return 0.5 * (self._C_eq * voltage**2)
    
    def energy2voltage(self, energy: float, SoC:float ) -> float:
        """
        Calcula tensão baseada na energia armazenada no banco de UC.
        :param float energy: Energia armazenada (J)
        :return float: Tensão do banco (V)
        """
        # return np.sqrt((2 * energy) / (self._C_eq))
        return sqrt((SoC * self._total_energy) / (50 * self._C_eq))
    

    def setCurrent(self, power: float) -> (float|float):
        """
        Calcula corrente baseada na potência requerida
        :param float power: Potência requerida (W)
        :return float i_sat: Corrente (A)
        :return float p_reject: Potência rejeitada (kW)
        """
        i = power / self._v_banco
        i_max = 280 * self._Np                                      # Corrente maxima no banco de UC
        i_sat = np.clip(i, -i_max, i_max)                           # Limita corrente em ambas direções
        i_reject = i - i_sat                                        # Calcula corrente rejeitada
        p_reject = (i_reject * self._v_banco) / 1000                # Calcula potência rejeitada
        return i_sat, p_reject


    def getTotalEnergy(self) -> float:
        """
        Retorna a energia total armazenada no banco de UC
        :return float: Energia total (J)
        """
        return self._total_energy
    
    
    def updateEnergy(self, current: float, dt: float) -> tuple[float, float, float, float]:
        """
        Atualiza energia do banco usando a corrente
        :param float current: Corrente do banco (A, + carga, - descarga)
        :param float dt: Intervalo de tempo (s)
        :return tuple[float, float]: (Tensão do banco, Energia armazenada)
        :return float p_reject: Potência rejeitada (kW)
        :return float i_uc: Verdadeira corrente do banco (A)
        """
        # Calcula variação de energia (P = V*I)
        energy_variation = -1 * self._v_banco * current * dt
        # print(f'DEBUG: energy_variation = {energy_variation}')
        
        # Calcula nova energia
        # print(f'DEBUG: self._stored_energy = {self._stored_energy}')
        # print(f'DEBUG: total_energy = {self._total_energy}')
        new_energy = self._stored_energy + energy_variation
        # print(f'DEBUG: new_energy = {new_energy}')
        
        # Limita energia baseado no SoC
        max_energy = self._total_energy * (self._SoC_max/100)
        # print(f'DEBUG: max_energy = {max_energy}')
        min_energy = self._total_energy * (self._SoC_min/100)
        # print(f'DEBUG: min_energy = {min_energy}')
        clip_energy = np.clip(new_energy, min_energy, max_energy)
        # print(f'DEBUG: clip_energy = {clip_energy}')
     
        
        p_reject = ((new_energy - clip_energy) / dt) / 1000         # Potência rejeitada (kW)
        i_reject = ((clip_energy - new_energy) / dt) / self._v_banco
        i_uc = current - i_reject
        # print(f'DEBUG: p_reject = {p_reject}')

        # print(f'DEBUG: i_reject = { i_reject}')
        # Preciso recalcular a corrente para considerar a potência rejeitada

        
        # Atualiza estados
        self._stored_energy = clip_energy
        self._SoC = (clip_energy / self._total_energy) * 100
        # print(f'DEBUG: self._SoC = {self._SoC}')
        self._v_banco = self.energy2voltage(clip_energy, self._SoC)
        # print(f'DEBUG: self._v_banco = {self._v_banco}')

        # print("-------------------------------------------------------------")
        # sleep(1)
        
        return self._SoC, self._v_banco, p_reject, i_uc