import torch

class BERTClass(torch.nn.Module):
    def __init__(self, model, n_class, n_nodes):
        super(BERTClass, self).__init__()
        self.l1 = model
        self.pre_classifier = torch.nn.Linear(n_nodes, n_nodes)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(n_nodes, n_class)
        #self.classifier = torch.nn.Linear(n_nodes, 512)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 512)
            #torch.nn.Linear(512, n_class),
            #torch.nn.Softmax(dim=1)
        )

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, n_class)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        #pooler = self.pre_classifier(pooler)
        #pooler = torch.nn.Tanh()(pooler)
        #pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        #output = pooler
        #print("language length")
        #print(output.size())
        return output