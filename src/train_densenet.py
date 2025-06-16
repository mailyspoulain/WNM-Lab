
#!/usr/bin/env python3
"""
Entraînement DenseNet121 pour détection de pneumonie
Version corrigée et optimisée pour ~1h d'entraînement
Compatible Windows avec multiprocessing
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

# ===== FONCTIONS D'ENTRAÎNEMENT (définies en dehors de main) =====
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_bar = tqdm(loader, desc='Training')
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        train_bar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{correct/total:.2%}'
        })
    
    return running_loss / total, correct / total

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = running_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return avg_loss, accuracy, auc, all_labels, all_preds, all_probs

def make_weights_for_balanced_classes(dataset):
    counts = np.bincount([label for _, label in dataset.samples])
    labels_weights = 1. / counts
    weights = labels_weights[[label for _, label in dataset.samples]]
    return torch.DoubleTensor(weights)

# ===== FONCTION PRINCIPALE =====
def main():
    # Fixer les seeds pour reproductibilité
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Utilisation de : {DEVICE}")

    # Chemins
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'chest_xray')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    OUTPUT_DIR = os.path.join('backend', 'models')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Hyperparamètres optimisés pour 1h d'entraînement
    BATCH_SIZE = 64 if torch.cuda.is_available() else 32
    EPOCHS = 8
    LR = 0.001
    PATIENCE = 2
    NUM_WORKERS = 0  # IMPORTANT: 0 sur Windows pour éviter l'erreur multiprocessing

    print("\n🚀 ENTRAÎNEMENT DENSENET121 - VERSION RAPIDE")
    print("=" * 60)

    # Transformations optimisées pour rapidité
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    print("📁 Chargement des datasets...")
    try:
        train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
        val_dataset = datasets.ImageFolder(VAL_DIR, transform=test_transform)
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
        
        print(f"Classes : {train_dataset.classes}")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        print("Vérifiez que les dossiers existent dans:", DATA_DIR)
        return

    # Équilibrage des classes
    weights = make_weights_for_balanced_classes(train_dataset)
    sampler = WeightedRandomSampler(weights, len(weights))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                             num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # ===== MODÈLE DENSENET121 =====
    print("\n🏗️ Construction du modèle DenseNet121...")

    # Créer le modèle avec l'architecture EXACTE qui sera utilisée dans model_loader_improved.py
    model = models.densenet121(weights='IMAGENET1K_V1')

    # IMPORTANT: Créer le même classifier que dans model_loader_improved.py
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    model = model.to(DEVICE)

    # Optimiseur et loss
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    # Loss avec poids pour gérer le déséquilibre
    class_counts = np.bincount([label for _, label in train_dataset.samples])
    class_weights = torch.tensor([1.0, class_counts[0]/class_counts[1]], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Poids des classes: Normal={class_weights[0]:.2f}, Pneumonie={class_weights[1]:.2f}")

    # ===== ENTRAÎNEMENT =====
    print("\n🎯 Début de l'entraînement...")
    start_time = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(OUTPUT_DIR, 'densenet121_chest_xray_model.pth')

    for epoch in range(EPOCHS):
        # Vérifier le temps écoulé
        elapsed = (time.time() - start_time) / 60
        if elapsed > 55:  # Arrêter à 55 minutes pour garder du temps pour l'évaluation
            print(f"\n⏰ Arrêt après {elapsed:.1f} minutes (limite de temps)")
            break
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{EPOCHS} - Temps écoulé: {elapsed:.1f} min")
        print(f"{'='*60}")
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validation
        val_loss, val_acc, val_auc, _, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)
        
        # Mise à jour du scheduler
        scheduler.step(val_acc)
        
        # Historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f"\n📊 Résultats Epoch {epoch+1}:")
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Sauvegarder UNIQUEMENT le state_dict pour compatibilité
            torch.save(model.state_dict(), best_model_path)
            
            # Sauvegarder aussi avec plus d'infos
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_auc': val_auc,
                'architecture': 'densenet121',
                'num_classes': 2,
                'class_names': train_dataset.classes,
                'history': history
            }
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'densenet121_checkpoint.pth'))
            
            print(f"   ✅ Nouveau meilleur modèle! Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ Pas d'amélioration (patience: {patience_counter}/{PATIENCE})")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping après {epoch+1} epochs")
            break

    # ===== TEST FINAL =====
    print("\n" + "="*60)
    print("📊 ÉVALUATION FINALE")
    print("="*60)

    # Charger le meilleur modèle
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    # Test
    test_loss, test_acc, test_auc, test_labels, test_preds, test_probs = evaluate_model(
        model, test_loader, criterion, DEVICE
    )

    print(f"\n🎯 Résultats sur le test set:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   AUC: {test_auc:.4f}")

    # Recherche du seuil optimal
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"   Seuil optimal: {optimal_threshold:.3f}")

    # Rapport de classification
    print("\n📋 Rapport de classification:")
    print(classification_report(test_labels, test_preds, 
                              target_names=['Normal', 'Pneumonie']))

    # Matrice de confusion
    cm = confusion_matrix(test_labels, test_preds)
    print("\n🔢 Matrice de confusion:")
    print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")

    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0

    print(f"\n   Sensibilité (détection pneumonie): {sensitivity:.2%}")
    print(f"   Spécificité (détection normal): {specificity:.2%}")

    # ===== GRAPHIQUES =====
    print("\n📈 Génération des graphiques...")

    plt.figure(figsize=(12, 4))

    # Courbes d'apprentissage
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss pendant l\'entraînement')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy pendant l\'entraînement')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Courbe ROC
    plt.subplot(1, 3, 3)
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
               label=f'Seuil optimal ({optimal_threshold:.3f})')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'), dpi=150)
    plt.close()

    # ===== CONFIGURATION FINALE =====
    print("\n💾 Sauvegarde de la configuration...")

    # Configuration pour model_loader_improved.py
    config = {
        'model_path': best_model_path,
        'optimal_threshold': float(optimal_threshold),
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_time': f"{(time.time() - start_time) / 60:.1f} minutes",
        'epochs_trained': len(history['train_loss']),
        'architecture_note': 'DenseNet121 avec classifier Sequential(Dropout, Linear, ReLU, Dropout, Linear)',
        'image_size': '224x224',
        'batch_size': BATCH_SIZE
    }

    with open(os.path.join(OUTPUT_DIR, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # ===== CODE À AJOUTER =====
    print("\n📝 IMPORTANT - Mettez à jour model_loader_improved.py:")
    print("-" * 60)
    print(f"""
Dans predict_pneumonia_improved(), modifier le seuil:
    threshold = {optimal_threshold:.3f}  # Seuil optimal trouvé
""")
    print("-" * 60)

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ ENTRAÎNEMENT TERMINÉ en {total_time:.1f} minutes!")
    print(f"   Modèle sauvé: {best_model_path}")
    print(f"   Configuration: {os.path.join(OUTPUT_DIR, 'model_config.json')}")
    print(f"   Graphiques: {os.path.join(OUTPUT_DIR, 'training_results.png')}")

    print("\n🎯 RÉSUMÉ DES PERFORMANCES:")
    print(f"   - Accuracy: {test_acc:.2%}")
    print(f"   - AUC: {test_auc:.3f}")
    print(f"   - Sensibilité: {sensitivity:.2%}")
    print(f"   - Spécificité: {specificity:.2%}")

    print("\n✨ Le modèle est prêt à être utilisé!")
    print("   Lancez 'python app.py' pour démarrer le serveur")
    print("   Puis testez avec 'python test_on_folder.py'")

# ===== POINT D'ENTRÉE PRINCIPAL - OBLIGATOIRE SUR WINDOWS =====
if __name__ == '__main__':
    main()