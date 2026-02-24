

clear; clc; close all;
rng(42); % Sonuçların her seferinde aynı çıkması için (Random Seed)

%% 1. VERİ YÜKLEME VE HAZIRLIK
disp('>>> Veri Yükleniyor...');

if ~isfile('oil_spill.csv')
    error('HATA: "oil_spill.csv" dosyası bulunamadı.');
end

opts = detectImportOptions('oil_spill.csv');
opts.VariableNamingRule = 'preserve';
data = readtable('oil_spill.csv', opts);

% Veriyi Ayır
X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));

% Eksik Veri Temizliği
if any(isnan(X), 'all') || any(isnan(y))
    disp('UYARI: NaN değerler temizleniyor...');
    idx = ~any(isnan(X), 2) & ~isnan(y);
    X = X(idx, :); y = y(idx);
end

% STRATIFIED SPLIT (Toolbox Fonksiyonu: cvpartition)
% Hedef değişkeni categorical yapıyoruz ki cvpartition hata vermesin
cv = cvpartition(categorical(y), 'HoldOut', 0.3);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% Normalizasyon (Z-Score)
[X_train_scaled, mu, sigma] = zscore(X_train);
X_test_scaled = (X_test - mu) ./ sigma;

disp(['Eğitim Seti: ' num2str(length(y_train))]);
disp(['Test Seti:   ' num2str(length(y_test))]);

%% SENARYO 1: TÜM ÖZELLİKLER
disp(' ');
disp('==========================================================');
disp('TABLO 1: TÜM ÖZELLİKLER İLE MODEL PERFORMANSI');
disp('==========================================================');

results1 = train_models_toolbox(X_train_scaled, y_train, X_test_scaled, y_test);
disp(struct2table(results1));

%% SENARYO 2: SEÇİLEN 10 ÖZELLİK
disp(' ');
disp('==========================================================');
disp('TABLO 2: SEÇİLEN 10 ÖZELLİK VE NORMALLİK TESTİ');
disp('==========================================================');

% Özellik İsimleri ve Seçimi
allVars = data.Properties.VariableNames;
targetFeats = {'f_47', 'f_1', 'f_2', 'f_3', 'f_25', 'f_46', 'f_6', 'f_48', 'f_35', 'f_32'};
featIdx = [];
for i=1:length(targetFeats)
    idx = find(strcmp(allVars, targetFeats{i}));
    if ~isempty(idx), featIdx = [featIdx idx]; end
end

% Normallik Testi (Toolbox Fonksiyonu: lillietest)
disp('--- Normallik Testi ---');
for i=1:length(featIdx)
    [h, p] = lillietest(X(:, featIdx(i)));
    if h==1, res='Normal Değil'; else, res='Normal'; end
    fprintf('%s : p=%.5f (%s)\n', targetFeats{i}, p, res);
end

X_train_sel = X_train_scaled(:, featIdx);
X_test_sel = X_test_scaled(:, featIdx);

disp(' ');
disp('--- Seçilen Özelliklerle Sonuçlar ---');
results2 = train_models_toolbox(X_train_sel, y_train, X_test_sel, y_test);
disp(struct2table(results2));

%% SENARYO 3: SMOTE
disp(' ');
disp('==========================================================');
disp('TABLO 3: SMOTE SONRASI PERFORMANS');
disp('==========================================================');

fprintf('SMOTE Öncesi: 0:%d, 1:%d\n', sum(y_train==0), sum(y_train==1));
[X_train_sm, y_train_sm] = apply_smote(X_train_scaled, y_train); % SMOTE hala manuel fonksiyon gerektirir
fprintf('SMOTE Sonrası: 0:%d, 1:%d\n', sum(y_train_sm==0), sum(y_train_sm==1));

disp(' ');
disp('--- SMOTE Uygulanmış Sonuçlar ---');
results3 = train_models_toolbox(X_train_sm, y_train_sm, X_test_scaled, y_test);
disp(struct2table(results3));


%% YARDIMCI FONKSİYONLAR

function results = train_models_toolbox(X_tr, y_tr, X_te, y_te)
    % Bu fonksiyon Toolbox fonksiyonlarını kullanır
    models = {'SVM', 'KNN', 'DecisionTree', 'MLP', 'RandomForest'};
    results = struct('Model', {}, 'Accuracy', {}, 'Precision', {}, 'Recall', {}, 'F1_Score', {}, 'ROC_AUC', {});
    
    for i = 1:length(models)
        name = models{i};
        try
            switch name
                case 'SVM'
                    mdl = fitcsvm(X_tr, y_tr, 'KernelFunction', 'rbf', 'Standardize', false);
                    [label, score] = predict(mdl, X_te);
                    probs = score(:,2); % Pozitif sınıf olasılığı (SVM için score transform gerekebilir ama fitcsvm probability default kapalıdır, burada score kullanıyoruz)
                    
                case 'KNN'
                    mdl = fitcknn(X_tr, y_tr, 'NumNeighbors', 5);
                    [label, score] = predict(mdl, X_te);
                    probs = score(:,2);
                    
                case 'DecisionTree'
                    mdl = fitctree(X_tr, y_tr);
                    [label, score] = predict(mdl, X_te);
                    probs = score(:,2);
                    
                case 'MLP'
                    % R2021a ve sonrası için fitcnet
                    if exist('fitcnet','file')
                        mdl = fitcnet(X_tr, y_tr, 'LayerSizes', 100);
                        [label, score] = predict(mdl, X_te);
                        probs = score(:,2);
                    else
                        % Eski versiyonlar için patternnet (Neural Network Toolbox)
                        net = patternnet(100);
                        net.trainParam.showWindow = false;
                        net = train(net, X_tr', dummyvar(categorical(y_tr))');
                        out = net(X_te');
                        [~, label] = max(out); label=label'-1; % 1-2'den 0-1'e çevir
                        probs = out(2,:)';
                    end
                    
                case 'RandomForest'
                    t = templateTree('Reproducible',true);
                    mdl = fitcensemble(X_tr, y_tr, 'Method', 'Bag', 'NumLearningCycles', 50, 'Learners', t);
                    [label, score] = predict(mdl, X_te);
                    probs = score(:,2);
            end
            
            % Metrikler
            confMat = confusionmat(y_te, label, 'Order', [0;1]);
            TP = confMat(2,2); TN = confMat(1,1); FP = confMat(1,2); FN = confMat(2,1);
            
            acc = (TP+TN) / sum(confMat(:));
            prec = TP / (TP+FP);
            rec = TP / (TP+FN);
            f1 = 2*prec*rec / (prec+rec);
            
            % NaN düzeltme
            if isnan(prec), prec=0; end
            if isnan(f1), f1=0; end
            
            % AUC
            try, [~,~,~,auc] = perfcurve(y_te, probs, 1); catch, auc=0.5; end
            
            results(i).Model = name;
            results(i).Accuracy = round(acc, 3);
            results(i).Precision = round(prec, 3);
            results(i).Recall = round(rec, 3);
            results(i).F1_Score = round(f1, 3);
            results(i).ROC_AUC = round(auc, 3);
            
        catch ME
            fprintf('Hata (%s): %s\n', name, ME.message);
            results(i).Model = name;
            results(i).Accuracy = 0;
        end
    end
end

function [X_res, y_res] = apply_smote(X, y)
    % MATLAB'da dahili SMOTE yoktur, bu yüzden bu manuel fonksiyonu kullanmaya devam etmelisiniz.
    idx1 = find(y==1); idx0 = find(y==0);
    X1 = X(idx1,:); X0 = X(idx0,:);
    n1 = length(idx1); n0 = length(idx0);
    N = n0 - n1;
    
    if N<=0, X_res=X; y_res=y; return; end
    
    syn = zeros(N, size(X,2));
    k = min(5, n1-1);
    
    for i=1:N
        r = randi(n1);
        base = X1(r,:);
        dists = sum((X1-base).^2, 2);
        [~, sort_idx] = sort(dists);
        nbr = X1(sort_idx(randi(k)+1), :);
        syn(i,:) = base + rand()*(nbr-base);
    end
    X_res = [X0; X1; syn];
    y_res = [zeros(n0,1); ones(n1,1); ones(N,1)];
end