import torch
import torch.nn as nn


class BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num):
		super(BPR, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""
		self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_item = nn.Embedding(item_num, factor_num)
		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, user, item_i, item_j):
		user = self.embed_user(user)
		item_i = self.embed_item(item_i)
		item_j = self.embed_item(item_j)

		prediction_i = (user * item_i).sum(dim=-1)
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j

class BPR_S(nn.Module):
	"""
	BPR for session recommendation.
	"""
	def __init__(self, item_num, factor_num):
		super(BPR_S, self).__init__()
		"""
		item_num: number of items;
		factor_num: number of predictive factors.
		"""
		# self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_item = nn.Embedding(item_num, factor_num)
		# nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, session, item_i, item_j):
		"""
		Args:
			session: list, [[item_id0, item_id1, item_id2,...]], batch_size * max_sess_len
			item_i: batch_size
			item_j: batch_size
		Returns:
		"""
		# user = self.embed_user(user)
		user = self.embed_item(session).mean(dim=1)
		item_i = self.embed_item(item_i)
		item_j = self.embed_item(item_j)

		prediction_i = (user * item_i).sum(dim=-1)
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j

if __name__ == "__main__":
	item_num = 10
	factor_num=10
	model = BPR_S(item_num, factor_num)

	session = torch.tensor([[0,1,2,3],[4,5,6,7],[8,9,2,3]])
	item_i = torch.tensor([4,5,6])
	item_j = torch.tensor([7,2,4])

	pre_i, pre_j = model(session,item_i,item_j)
	print(pre_i)
	print(pre_j)