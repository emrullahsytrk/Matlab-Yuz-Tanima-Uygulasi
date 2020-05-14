load('gTruth.mat')
load('Detector.mat')
denemeGTruth = selectLabels(gTruth,'Emrullah');

img='C:\Users\Emrullah\Desktop\Örüntü Tanýma\Yüz Tanýma\35.jpg';
a=imread(img);

trainingData = objectDetectorTrainingData(denemeGTruth,'SamplingFactor',2,...
'WriteLocation','TrainingData');

detector = trainACFObjectDetector(trainingData,'NumStages',168);
save('Detector.mat','detector');

i = 1;

results = struct('Boxes',[],'Scores',[]);
  [bboxes, scores] = detect(detector,a,'Threshold',1);
 
%   Select strongest detection 
    [~,idx] = max(scores);
    results(i).Boxes = bboxes;
    results(i).Scores = scores;

%   VISUALIZE
    annotation = sprintf('%s , Confidence %4.2f',detector.ModelName,scores(idx));
    I = insertObjectAnnotation(a,'rectangle',bboxes(idx,:),annotation);

  figure, imshow(I)
  