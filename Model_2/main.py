import numpy as np
import matplotlib.pyplot as plt
import auxiliary as aux
from copy import copy
import Particle_Thompson_Sampling as PTS


np.set_printoptions(precision=4)



if __name__ == "__main__":
    
    # Set up model parameters
    D = 3        # number of domains
    B = [2,3,3]  # number of resource blocks in each domain
    T = 1000      # time horizon
    
    N_simul = 500   # number of simulations
    
    AVG_REG10_PTS1 = np.zeros(T)   # average regret
    AVG_REG100_PTS1 = np.zeros(T)   # average regret
    AVG_REG500_PTS1 = np.zeros(T)   # average regret
    AVG_REG10_PTS2 = np.zeros(T)   # average regret
    AVG_REG100_PTS2 = np.zeros(T)   # average regret
    AVG_REG500_PTS2 = np.zeros(T)   # average regret    
    
    # run simulations
    for i in range(N_simul):
        if i % 1 == 0:
            print('Simulation', i)
        
        G10_PTS1 = PTS.System_PTS1(D, B, T, Npar_sys=10)
        G10_PTS1.init_true_parameter()
        G10_PTS1.init_particles()
        G10_PTS1.run()
        AVG_REG10_PTS1 += G10_PTS1.AVG_REG    
    
        G100_PTS1 = PTS.System_PTS1(D, B, T, Npar_sys=100)
        G100_PTS1.init_true_parameter()
        G100_PTS1.init_particles()
        G100_PTS1.run()
        AVG_REG100_PTS1 += G100_PTS1.AVG_REG
        
        G500_PTS1 = PTS.System_PTS1(D, B, T, Npar_sys=500)
        G500_PTS1.init_true_parameter()
        G500_PTS1.init_particles()
        G500_PTS1.run()
        AVG_REG500_PTS1 += G500_PTS1.AVG_REG        
        
        G10_PTS2 = PTS.System_PTS2(D, B, T, Npar_blk=10)
        G10_PTS2.init_true_parameter()
        G10_PTS2.init_particles()
        G10_PTS2.run()
        AVG_REG10_PTS2 += G10_PTS2.AVG_REG    
    
        G100_PTS2 = PTS.System_PTS2(D, B, T, Npar_blk=100)
        G100_PTS2.init_true_parameter()
        G100_PTS2.init_particles()
        G100_PTS2.run()
        AVG_REG100_PTS2 += G100_PTS2.AVG_REG        

        G500_PTS2 = PTS.System_PTS2(D, B, T, Npar_blk=500)
        G500_PTS2.init_true_parameter()
        G500_PTS2.init_particles()
        G500_PTS2.run()
        AVG_REG500_PTS2 += G500_PTS2.AVG_REG        
        
        
    AVG_REG10_PTS1 = AVG_REG10_PTS1 / N_simul
    AVG_REG100_PTS1 = AVG_REG100_PTS1 / N_simul
    AVG_REG500_PTS1 = AVG_REG500_PTS1 / N_simul
    AVG_REG10_PTS2 = AVG_REG10_PTS2 / N_simul
    AVG_REG100_PTS2 = AVG_REG100_PTS2 / N_simul
    AVG_REG500_PTS2 = AVG_REG500_PTS2 / N_simul
    
    # plot  
    plt.figure(1) 
    plt.plot(range(T), AVG_REG10_PTS1, 'k-', marker='o', markevery=100, label='10 per-system particles')
    plt.plot(range(T), AVG_REG100_PTS1, 'k-', marker='^', markevery=100, label='100 per-system particles')
    plt.plot(range(T), AVG_REG500_PTS1, 'k-', marker='s', markevery=100, label='500 per-system particles')    
    plt.plot(range(T), AVG_REG10_PTS2, 'k--', marker='o', markevery=100, label='10 per-block particles')
    plt.plot(range(T), AVG_REG100_PTS2, 'k--', marker='^', markevery=100, label='100 per-block particles')
    plt.plot(range(T), AVG_REG500_PTS2, 'k--', marker='s', markevery=100, label='500 per-block particles')    
    plt.legend()
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('running average regret')
    plt.show()   
     
    
    



      

       
