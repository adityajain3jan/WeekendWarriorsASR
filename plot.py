import matplotlib.pyplot as plt

ctc_file = open("ctc_results.txt")
wctc_file = open("wctc_results.txt")


ctc_lines  = ctc_file.readlines()
wctc_lines = wctc_file.readlines()

ctc_losses = []
ctc_wer = []

for line in ctc_lines:
    ctc_losses.append(float(line.replace(',', ' ').split()[1]))
    ctc_wer.append(float(line.replace(',', ' ').split()[3]))
    
    
wctc_losses = []
wctc_wer = []

for line in wctc_lines:
    wctc_losses.append(float(line.replace(',', ' ').split()[1]))
    wctc_wer.append(float(line.replace(',', ' ').split()[3]))
  
bound = min(len(ctc_lines), len(wctc_lines))
plt.plot(list(range(len(ctc_losses))), ctc_losses, label='CTC Loss')
plt.title("Normal CTC Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("ctc_losses.png") 

plt.clf()
plt.plot(list(range(len(ctc_losses))), ctc_wer, label='CTC WER')
plt.title("Normal CTC WER vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("WER")
plt.savefig("ctc_wer.png") 


plt.clf()
plt.plot(list(range(len(wctc_losses))), wctc_losses, label='WCTC Loss')
plt.title("WCTC Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("wctc_losses.png") 
plt.clf()

plt.plot(list(range(len(wctc_wer))), wctc_wer, label='WCTC WER')
plt.title("WCTC WER vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("WER")
plt.savefig("wctc_wer.png") 
