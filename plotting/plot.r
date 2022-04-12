#! /.mounts/labs/simpsonlab/sw/miniconda3/envs/cfdna/bin/Rscript
library(ggplot2)
data = 'results/h32.tsv'

df = read.table(data, header=TRUE)
print(df)

plot = ggplot(df, aes(K, Accuracy, fill=Legend))+
    geom_col(position='dodge')+
    labs(title = "Graph Attention Network Accuracy on ADReSS 2020 Test Set (32d)",
         x = "Number of layers (K)",
         y = "Classification Accuracy")+
    xlim(0,10) +
    scale_x_continuous(breaks=c(0,1,2,3,4,5,6,7,8,9))+
    ylim(0.4, 1)+
    scale_y_continuous(breaks=c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))

ggsave('plots/h32.png')
