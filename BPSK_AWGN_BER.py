import numpy as np 
from scipy import special
import matplotlib.pyplot as plt


# ----- Parameter Definition ----------------------------------
Eb_uncoded_BPSK = 1
Eb_Hamming74_BPSK = 1
Navg = 80000



# ----- (7,4) Hamming code -----------------------------------------
k = 4
n = 7
Rc = k/n

G = np.array([[1, 0, 0, 0, 1, 1, 0],
              [0, 1, 0, 0, 0, 1, 1],
              [0, 0, 1, 0, 1, 1, 1],
              [0, 0, 0, 1, 1, 0, 1]])

all_possible_messege = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 0, 1, 1],
                                 [0, 1, 0, 0],
                                 [0, 1, 0, 1],
                                 [0, 1, 1, 0],
                                 [0, 1, 1, 1],
                                 [1, 0, 0, 0],
                                 [1, 0, 0, 1],
                                 [1, 0, 1, 0],
                                 [1, 0, 1, 1],
                                 [1, 1, 0, 0],
                                 [1, 1, 0, 1],
                                 [1, 1, 1, 0],
                                 [1, 1, 1, 1]])

codeword = (all_possible_messege.dot(G)) % 2

# ----- SNR computation range-----------------------------------------------
gamma_dB = np.arange(-3,11,0.5)               # Eb/No (dB)
gamma_lin = np.power(10,gamma_dB/10)          # Eb/No (linear)
gamma_dB_Hamm74 = gamma_dB-10*np.log10(k/n)



# ----- AWGN Noise Standard Deviation --------------------------------------
sigma_w = (Eb_uncoded_BPSK/(2*gamma_lin))**0.5;



# ----- BER BPSK Uncoded Theory --------------------------------------------
BER_uncoded_BPSK_AWGN_Theory = 0.5*special.erfc((gamma_lin)**0.5)


# ----- BER (7,4) Hamming Code Theory --------------------------------------
BER_74_Hamming_Theory = (3/2)*special.erfc(((12/7)*gamma_lin)**0.5)


# ----- Initialization ----------------------------------------------------
hamm_d = np.zeros([len(sigma_w),2**k])
corr = np.zeros([len(sigma_w),2**k])
BER_Hamm_Hard = np.zeros([len(sigma_w),1])
BER_Hamm_Soft = np.zeros([len(sigma_w),1])



for m in range(len(sigma_w)):
    print('SNR = ', gamma_dB[m],'(dB)')
    total_errors_hard = 0.0
    total_errors_soft = 0.0
    
    for simu_iter in range(Navg):
    
        # ========== Tansmitter =====================================
        message = np.random.randint(0,2,[1,k])
        coded_message = (message.dot(G)) % 2
        # BPSK Modulation
        BPSK_coded_message = np.where(coded_message==0, -1, 1)
        
        # ========== AWGN Channel ===================================
        noise = sigma_w[m]*np.random.randn(1,n)
        BPSK_coded_message_noise = BPSK_coded_message+noise
        
        
        # ========== Hard decision ===================================
        hard_decision = np.where(BPSK_coded_message_noise>0, 1, 0)
      
        for i in range(2**k):
            hamm_d[m,i] = np.count_nonzero(hard_decision != codeword[i,:])
        
        pos_hard = np.argmin(hamm_d[m,:])
        est_hard = codeword[pos_hard,:]
        
        # Bit Error Count
        new_errors_hard = np.count_nonzero(est_hard != coded_message)
        total_errors_hard = total_errors_hard + new_errors_hard
        
        
        
        # ========== Soft decision ===================================
        for i in range(2**k):
            corr[m,i] = np.sum(BPSK_coded_message_noise*codeword[i,:])
        
        pos_soft = np.argmax(corr[m,:])
        est_soft = codeword[pos_soft,:]
        
        # Bit Error Count
        new_errors_soft = np.count_nonzero(est_soft != coded_message)
        total_errors_soft = total_errors_soft + new_errors_soft

        
    # ========== Bit Error Rate ===================================  
    BER_Hamm_Hard[m] = total_errors_hard/(Navg*n)
    BER_Hamm_Soft[m] = total_errors_soft/(Navg*n)
    
    
    
    
        
        
        
        
# ----- Figure Plot -------------------------------------
plt.semilogy(gamma_dB, BER_uncoded_BPSK_AWGN_Theory)
plt.semilogy(gamma_dB, BER_74_Hamming_Theory)
plt.semilogy(gamma_dB_Hamm74, BER_Hamm_Hard,ls='None',marker='o')
plt.semilogy(gamma_dB_Hamm74, BER_Hamm_Soft,ls='None',marker='*')
plt.title('BPSK, AWGN Channel')
plt.xlim(1, 9)
plt.xlabel('Eb / No (dB)')
plt.ylim(10**-7, 10**0)
plt.ylabel('Bit Error Probability')
plt.legend(('Uncoded BPSK Theory',\
            '(7,4) Hamming Code Theory, Soft Decision (1st term)',\
            '(7,4) Hamming Code Simulation, Hard Decision',\
            '(7,4) Hamming Code Simulation, Soft Decision'))
plt.grid(True)
plt.show()