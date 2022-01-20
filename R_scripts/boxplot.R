library(tidyverse)
library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(viridis)
df = read.csv('~/Downloads/mdf.csv')
df = df %>%
  separate(type, into = c('models','feature_slet_type'), sep = '[_]')
#boxplot(acc~models/feature_slet_type, df)

feature_slet_type = df[['feature_slet_type']]
feature_slet_type[feature_slet_type == 'a'] = 'Select on \n train + validation'
feature_slet_type[feature_slet_type == 'nfs'] = 'No feature selection'
feature_slet_type[feature_slet_type == 'p'] = 'Select on train'
df$fs = feature_slet_type

df$models[df$models == 'GBM'] = 'GB'
df$models[df$models == 'XGB']= 'XGBoost'
p = ggplot(data = df, aes(x= models, y = acc, fill = models))+
  geom_violin(width = 0.6, alpha = 0.5)+
  facet_wrap(~ fs)+
  geom_boxplot(width=0.2, color="black", alpha=0.2)+
  scale_fill_viridis(discrete = TRUE) +
  theme_bw()+
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.title = element_text(size = 12),
        strip.text = element_text(size = 13))+
  ylab('Accuracy score')+
  guides(fill=guide_legend(title="Models", face = 'bold'))

pdf('~/jerry_jupyter/CATS/processed/graphes/Boxplot.pdf', width = 8, height = 4)
print(p)
dev.off()


# statistical test. 

sta_test = function(group, model_1, model_2) {
  data = df[df[["feature_slet_type"]] == group,]
  acc_1 = data[data[["models"]] == model_1,]$acc
  acc_2 = data[data[['models']] == model_2,]$acc
  
  print(sd(acc_1))
  print(sd(acc_2))
  t.test(acc_1, acc_2)
}





