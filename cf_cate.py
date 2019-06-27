# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   协同过滤  计算cate相似度矩阵
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def CateSimilarity(actions):
	C = dict()
	N = dict()
	for user, cate in actions.items():
		for i in cate:
			N[i] += 1
			for j in cate:
				if i==j:
					continue
				C[i][j] += 1
	# 计算相似度矩阵
	W = dict()
	for i, related_cates in C.items():
		for j, c_ij in related_cates.items():
			W[i][j] = c_ij / math.sqrt(N[i]*N[j])
	return W

def CateSimilarity_2(actions):
	C = dict()
	N = dict()
	for u, items in actions.items():
		for i in users