import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

df = pd.read_csv('classification_trainset.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Logistic Regression
log_reg = LogisticRegression()
log_reg_scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy')
print("Logistic Regression CV Accuracy: {:.2f}%".format(log_reg_scores.mean() * 100))

# 2. Decision Tree
tree = DecisionTreeClassifier()
tree_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring='accuracy')
print("Decision Tree CV Accuracy: {:.2f}%".format(tree_scores.mean() * 100))

# 3. Random Forest
forest = RandomForestClassifier()
forest_scores = cross_val_score(forest, X_train, y_train, cv=5, scoring='accuracy')
print("Random Forest CV Accuracy: {:.2f}%".format(forest_scores.mean() * 100))

# 4. Gradient Boosting
gboost = GradientBoostingClassifier()
gboost_scores = cross_val_score(gboost, X_train, y_train, cv=5, scoring='accuracy')
print("Gradient Boosting CV Accuracy: {:.2f}%".format(gboost_scores.mean() * 100))

# 5. XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='accuracy')
print("XGBoost CV Accuracy: {:.2f}%".format(xgb_scores.mean() * 100))

# 6. SVM
svm = SVC()
svm_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
print("SVM CV Accuracy: {:.2f}%".format(svm_scores.mean() * 100))

# 7. Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier()),
    ('svc', SVC(probability=True)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
], voting='soft')
voting_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
print("Voting Classifier CV Accuracy: {:.2f}%".format(voting_scores.mean() * 100))

# 8. Stacking Classifier
estimators = [
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier()),
    ('svc', SVC(probability=True)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_scores = cross_val_score(stacking_clf, X_train, y_train, cv=5, scoring='accuracy')
print("Stacking Classifier CV Accuracy: {:.2f}%".format(stacking_scores.mean() * 100))

# 9. Deep Neural Network (PyTorch)
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0.1):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        residual = x
        x = self.relu(self.layer1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = x + residual
        x = self.relu(x)
        return x
    
class ResidualNN(nn.Module):
    def __init__(self):
        super(ResidualNN, self).__init__()
        self.layer1 = nn.Linear(10, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.residual_block1 = ResidualBlock(16, 16)
        self.residual_block2 = ResidualBlock(16, 16)
        self.residual_block3 = ResidualBlock(16, 16)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.bn1(self.layer1(x))
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.sigmoid(self.output(x))
        return x

# 모델 초기화 및 손실 함수, 옵티마이저 정의
model = ResidualNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# optimizer = optim.SGD(model.parameters(), lr=0.005)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Evaluate on validation set
X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
y_val_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

train_loss_history = []
val_loss_history = []
best_val_accuracy = -float('inf')
best_val_loss = float('inf')
best_model_state_dict = None

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    val_accuracy = 100 * correct / total
    
    if val_accuracy >= best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state_dict = model.state_dict()
        best_epoch = epoch + 1
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()
            best_epoch = epoch + 1

    # if (epoch + 1) % 5 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

best_model_filename = f'best_model_{best_val_loss:.4f}_{best_epoch}_{best_val_accuracy:.2f}.pth'
torch.save(best_model_state_dict, best_model_filename)

plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_loss_history, label='Train Loss')
plt.plot(range(num_epochs), val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss_plot.png')