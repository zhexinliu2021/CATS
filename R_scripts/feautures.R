## feauture plot. 

GB_df = read.csv('~/Downloads/未命名文件夹/GBN_fea.csv')
XGB_df = read.csv('~/Downloads/未命名文件夹/XGB_fea_cou_part.csv')
knn_df = read.csv('~/Downloads/未命名文件夹/sorted_filc.csv')
rf_df = read.csv('~/Downloads/未命名文件夹/rf_fea_cou.csv')

rename = function(a){
  colnames(a) = c('id','num')
return(a)
  }

GB_df = rename(GB_df); XGB_df = rename(XGB_df);knn_df = rename(knn_df); rf_df = rename(rf_df)

GB_df$model = rep('kNN', )
get_100 = function(a){
  return(a[1:50,])
}
GB_df = get_100(GB_df);XGB_df = get_100(XGB_df); knn_df = get_100(knn_df); rf_df = get_100(rf_df)


GB_df$model = rep('GB', dim(GB_df)[1]);XGB_df$model = rep('XGB', dim(XGB_df)[1]);
knn_df$model = rep('kNN', dim(knn_df)[1]); rf_df$model = rep('RF', dim(rf_df)[1])

df = rbind(GB_df, XGB_df, knn_df, rf_df)

ggplot(data = df, aes(x = id, y = num, col = model))+
  geom_col(position = 'identity')


##
gb = data.frame(ID = GB_df[,1], gb = GB_df[,2])
knn = data.frame(ID = knn_df[,1], knn = knn_df[,2])
rf= data.frame(ID = rf_df[,1], rf = rf_df[,2])
xgb = data.frame(ID = XGB_df[,1], xgb = XGB_df[,2])

merge_df = rf %>%
  full_join(gb, by = 'ID') %>%
  full_join(xgb, by = 'ID') %>%
  full_join(knn, by = 'ID')


merge_df$rank = 1:dim(merge_df)[1]
merge_df = merge_df[1:50,]

check_1 = function(row){
  re = 0
  if (sum(!is.na(row[c('rf','gb','xgb','knn')] ) ) == 1 ){
    re = -20} 
  else {re = 0}
  return(re)
}
check_2 = function(row){
  re = 0
  if (sum(!is.na(row[c('rf','gb','xgb','knn')] ) ) == 2 ){
    re = -20} 
  else {re = 0}
  return(re)
}
check_3 = function(row){
  re = 0
  if (sum(!is.na(row[c('rf','gb','xgb','knn')] ) ) == 3 ){
    re = -20} 
  else {re = 0}
  return(re)
}
check_4 = function(row){
  re = 0
  if (sum(!is.na(row[c('rf','gb','xgb','knn')] ) ) == 4 ){
    re = -20} 
  else {re = 0}
  return(re)
}
merge_df$one = apply(merge_df, 1, FUN = check_1)
merge_df$two = apply(merge_df, 1, FUN = check_2)
merge_df$three =  apply(merge_df, 1, FUN = check_3)
merge_df$four = apply(merge_df, 1, FUN= check_4)

m_df = merge_df

merge_df =   gather(merge_df, key = model, value = counts, c(2:5,7:10))


pic = ggplot(merge_df, aes(x = rank, y = counts, fill = model))+
  geom_col(position = 'identity', alpha = 1,, color = NaN)+
  scale_color_manual(values = c('one'='#FFFF00', 'two'='#EEEE00', 'three'='#CDCD00', 'four' = '#8B8B00',
                                'gb' = '#1B9E77','rf'='#D95F02', 'xbg' = '#7570B3','knn' = '#E7298A'),
                     aesthetics = c( "fill"), labels = c('gb' ='GB',"knn" = 'kNN',"rf" = 'RF',"xbg" = 'XGB',"one" = 'Occur in one model',"two"='Occur in two models',
                                                        "three" = 'Occur in three models',"four" = 'Occur in four models'), name = 'Models and feature\noccurrence heatmap',
                     limits = c('gb','knn','rf','xbg','one','two','three','four'))+
  theme_bw()+
  xlab('Features ranked according to occurence')+
  ylab('Counts')+
  theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size= 15),
        legend.title = element_text(size = 13), strip.text = element_text(size = 20))

pdf("~/jerry_jupyter/CATS/processed/graphes/im_features.pdf",height = 5, width = 10)
print(pic)
dev.off()


###
m_df_top = m_df[1:15,]
m_df_rest = m_df[16:50,]


m_df_top =   gather(m_df_top, key = model, value = counts, c(2:5))
m_df_rest = gather(m_df_rest, key = rest_model, value = rest_counts, c(2:5))
colnames(m_df_rest) = c('rest_id','rest_rank','rest_model','rest_counts')

final_df = merge(m_df_top,m_df_rest)

picture = ggplot(data = final_df, aes(x = ID, y = counts ))+
  geom_col(color  = 'red', fill = 'red', position = 'identity', width =0.5)+
  
  geom_bar( aes(x = rest_id, y = rest_counts),stat='identity', position = position_identity(),
            color = '#FF6347',fill = '#FF6347', alpha = 0.3)+
  theme_gray()+
  xlab('')+
  ylab('Counts')+
  theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size= 15),
        legend.title = element_text(size = 13), strip.text = element_text(size = 20))


pdf("~/jerry_jupyter/CATS/processed/graphes/features_to_chr.pdf",height = 3, width = 10)
print(picture)
dev.off()
  


