import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class TransformerClassifier(nn.Module):
    def __init__(self, n_features):
        super(TransformerClassifier, self).__init__()
        self.transformer = nn.Transformer(nhead=8, num_encoder_layers=12)
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.transformer(x)
        x = self.linear(x)
        return x

# Assume we have some data
n_samples = 100
n_features = 30
X = torch.randn(n_samples, n_features)
y = torch.randint(0, 2, (n_samples, 1)).float()

# Initialize the model
model = TransformerClassifier(n_features)

# Define the loss function and the optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))
         # Evaluation
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            loss_val = criterion(outputs_val, y_val)
        print ('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'
               .format(epoch+1, 100, loss.item(), loss_val.item()))
        model.train()
