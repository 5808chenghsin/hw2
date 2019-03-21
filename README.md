# hw2

ISA525700 Computer Vision for Visual Effects<br/>Assignment 2: MUNIT<br/>Team 19
===

## Abstract
現今3C資訊產品市場規模龐大，消費者對於電腦視覺品質要求，亦日益升高。本文所探討MUNIT model具有一對多影像輸出轉換效果，對於其轉換後影像品質與多樣性等層面效果如何?這是在本文與實驗中欲深入探討的。

Keyword: 電腦視覺, MUNIT, 影像品質, 多樣性

## Table of Contents
1. [Abstract](#Abstract)
2. [Introduction](#Introduction)
3. [Training Process](#Training-Process)
4. [Inference Munit in personal image](#Inference-Munit-in-personal-image)
5. [Other Methods](#Other-Methods)
6. [Reference](#Reference)

## Introduction

- 現今全球電腦、平板與智慧型手機等3C資訊產品普及，消費者對電腦視覺的品質與多樣性，亦日益增高。在手機方面，根據[IDC 2018年第3季調查報告](https://3c.ltn.com.tw/news/35026)指出，全球智慧型手機總出貨量約3.5億支，市場規模龐大。

- 因UNIT model僅能一對一影像轉換，因此Xun Huang等四位學者在2018年Cornell University與NVIDIA的產學合作計劃中，發表提出MUNIT model，成功突破此難題，挑戰一對多影像生成，成功轉出多張影像。而本文正探討MUNIT的影像品質與多樣性等電腦視覺效果，與其它ML Model比較是否具有提供消費者最佳品質。

- 讓我們觀看下面關於MUNIT實驗成果影片。
[![](http://img.youtube.com/vi/ab64TWzWn40/0.jpg)](http://www.youtube.com/watch?v=ab64TWzWn40 "")

- 因為電腦視覺(Computer Vision)領域中，從ㄧ個影像從ㄧ處擷取轉移到另一處，存有許多問題，像是高強度辨識率、著色、修復、屬性移轉、樣式移轉等。像這種跨域的影像對影像轉換(cross-domain image-to-image translation)的問題，目前受到全球重大關注! 當dataset 與paired examples進行mapping時，可以透過條件生成模型(conditional generative model) 或簡單生成模型(simple regression model)加以解決。在unsupervision下，我們更關注對於影像內容環境的挑戰。在跨域配對(cross-domain mapping)領域上，多模態(multimodal)最令人感到興趣，由於影像中天氣、時間、光照等原因，現今技術採確定性(deterministic)或單峰(unimodal)進行mapping。結果圖片仍然無法全部分佈而輸出，縱使加入noise，深度學習神經網路仍舊無法learning。

- 因此本文提出Xun Huang的MUNIT，首先我們假設影像的潛在空間，可以被分解為內容碼(content code)與樣式碼(style code)，進一步讓兩張不同影像共享content code，但不共享style code，然後兩張影像重新組合訓練後，模型產生多樣性與多模態輸出，兼具高影像品質，取代傳統方法。

## Method
MUNIT是源自UNIT與Cycle GAN而來，目標是具有Bicycle GAN輸出一對多的功能，但只要有 unpaired instances 即可進行訓練，相關基礎概念說明如下。

### UNIT
非監督式影像對影像轉換法(Unsupervised Image-to-Image Translation, UNIT)，可以對兩個dataset進行Training，如老虎與獅子兩種照片，可以深度學習到眼睛、鼻子、嘴等等五官之共同特徵。

### Bicycle GAN
Bicycle GAN為監督式學習(supervised learning)，經過pair instances即可訓練出共同特徵，再透過亂數(Random)改變輸出，可以生成一對多。

### MUNIT
- 本文探討多模態非監督式影像對影像轉換法(Multimodal Unsupervised Image-to-Image Translation, MUNIT)。此模型主要透過內容碼(content code)與樣式碼(style code)學習圖片生成。

- 多模態(Multimodal)概念為生成一對多圖片風格轉換(style transfer)，例如將老虎的圖片轉成多張獅子圖片。Multimodal有隨機模型(stochastic model)與確定型模型(deterministic model)兩種。隨機模型(stochastic model)是輸入一張圖片可輸出多張隨機不同圖片。確定型模型(deterministic model)是一對一固定輸出圖片。

- 非監督式(Unsupervised)為將兩個不相同dataset一起訓練，例如老虎與狗照片，學習牠們五官特徵。

- 內容碼(content code)是學習共同特徵，例如將老虎的臉部五官特徵學習出，變成多張獅子照片。風格碼(style code)功能為將原圖轉換多張不同風格的圖片，例如秋天原圖轉成春季影像。如下圖所示，X1為輸入圖片，X2輸出圖片，C為content code，S1與S2為style code，X2是依據S2亂數產生多張不同圖片。

![](https://i.imgur.com/nb7pHx0.png)
 
- 如果想將一張影像風格套用在另一張影像的內容，我們可以從深度學習網路中擷取內容與風格兩種特徵，再將兩者進行重建，並採用損失函數與最佳演算法找出最佳影像輸出。


### Loss Function
![](https://i.imgur.com/QPyuVNK.png)

上述loss function 可拆成兩部份，GAN本身的 ![](https://i.imgur.com/0Uijq6h.gif) 加上reconstruction loss。


- GAN 的 ![](https://i.imgur.com/0Uijq6h.gif): 
該部份的loss即是傳統GAN的loss，主要目地在於讓Discriminator能有分出Generator產生的圖片和真正的圖片 ![](https://i.imgur.com/50HytWx.gif) ，同時讓 Generator 產生的圖片越來越逼真而能夠騙過 Discriminator 。

- reconstruction loss:
共可分為以下三種形式
    - ![](https://i.imgur.com/079YaZm.png)
    > 第一種是讓content encoder，以及style encoder將原圖分別enode後，將這些encode後的資訊拿來給GAN產生圖片，和原圖相同
    - ![](https://i.imgur.com/3PBZe5e.png)
    > 第二種是希望content code能在decode和encode過程中保留(EX: 狗轉貓後，貓和狗的五官(content code)希望都保留下來且是相同的)
    - ![](https://i.imgur.com/amew9hy.png)
    > 第二種是希望style code能在decode和encode過程中保留(狗轉某特定貓後，能夠用decoder得知該特定貓的品種(style code))

## Training Process
如下圖所示，MUNIT影像訓練，一個cycle單位為一萬。

![](https://i.imgur.com/K0fJtRy.jpg)

### Simulation with App
---
風格轉換的App市面上也已經出現不少，效果也都很好，本文先以App對新竹轉運站作為模擬電腦視覺效果，結合夕陽美景(Style)，呈現出燈會美景。

| _Source_  | _Style_  |  Transfer |
|:-----:|:----:|:----:|
|![](https://i.imgur.com/96QDBcW.jpg)|![](https://i.imgur.com/HSvqfms.jpg)|![](https://i.imgur.com/B3C2W0F.jpg)|![]
|新竹轉運站的原圖|以夕陽照片為訓練|創作出燈會美景|


## Inference Munit in personal image

### 1.Pick 3 different paintings
---
訓練完成該畫風的模型後，選其中三張畫作做為 style 參考，選三張真實照片(一張自己提供兩張由datasets隨機選出)分別就不同 style 進行轉換，其中公園的真實照片轉成 style2 的畫作最為成功!判斷是因為其情境十分相似，但普遍效果不佳，所以決定直接產生 style code 來對照片進行轉換。
#### a.Photo2Monet
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/NuFTDkZ.jpg)|![](https://i.imgur.com/lVyF84M.jpg)|![](https://i.imgur.com/VqONRxd.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/Nl6BLum.jpg)|![](https://i.imgur.com/MuX5MPi.jpg)|![](https://i.imgur.com/JqIo6wT.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/Ox2UAMS.jpg)|![](https://i.imgur.com/JyCTh4H.jpg)|![](https://i.imgur.com/htA5CZa.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/9BYzAiP.jpg)|![](https://i.imgur.com/WfsSU6A.jpg)|![](https://i.imgur.com/98YEb3R.jpg)|


#### b.Photo2Vangogh
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/effAhcg.jpg)|![](https://i.imgur.com/GJJVWgi.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/oIEitHE.jpg)|![](https://i.imgur.com/IpX6YcI.jpg)|![](https://i.imgur.com/2oCwpD2.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/R79DEUA.jpg)|![](https://i.imgur.com/TVu67O3.jpg)|![](https://i.imgur.com/hLnxhrY.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/L6QT7OJ.jpg)|![](https://i.imgur.com/WIiRck6.jpg)|![](https://i.imgur.com/rUkSHs4.jpg)|

#### c.Photo2Cezanne 
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/KdgdtLu.jpg)|![](https://i.imgur.com/MYFTUBs.jpg)|![](https://i.imgur.com/YQ8HW1G.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/IcT2iE6.jpg)|![](https://i.imgur.com/Dqze08V.jpg)|![](https://i.imgur.com/QEEpV67.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/dBGmUTn.jpg)|![](https://i.imgur.com/E6iEhqh.jpg)|![](https://i.imgur.com/V3aACNP.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/08ecHL8.jpg)|![](https://i.imgur.com/ZhRmWZE.jpg)|![](https://i.imgur.com/zRuMIxI.jpg)|
### 2.Linearity produce style code
---
自己產生 style code 進行轉換我們使用了兩種技巧，一種是技巧A,隨機產生兩個 style code 並以兩者變化量除16做為單位向量，此向量有8個維度，因為儲存 style_dim 在訓練時設定為8，技巧B則是一次只動 style code 的其中一個維度，範圍從-3.5到4，遞增0.5，其他維度設為0。
#### 技巧A

向量調整style code
![](https://i.imgur.com/1i8H6P0.jpg)

|![](https://i.imgur.com/D1KCYtB.gif)|![](https://i.imgur.com/94PG8rX.gif)| ![](https://i.imgur.com/GbdkQii.gif)|
|:--:|:--:|:--:|
|cezanne|monet|vangogh|

因為是用兩個隨機的style code產生單位向量，所以每個維度變化尺度不同，畫風線性變化上不一定很明顯，不過產生的圖片在風格上很接近畫作相較於直接抽取畫作的 style code。
#### 技巧Ｂ
之所以維度調整範圍由-3.5到4是因為幾乎每個維度在這個範圍內都有明顯的線性變化。




|dim|Monet|Vangogh|Cezanne|Mixdataset|
|:--:|:--:|:--:|:--:|:--:| 
|1|![](https://i.imgur.com/1TIYaZQ.gif)|![](https://i.imgur.com/1ckiYCd.gif)|![](https://i.imgur.com/M1HWfNd.gif)|![](https://i.imgur.com/UI1WbBk.gif)|
|2|![](https://i.imgur.com/A6mxQBD.gif)|![](https://i.imgur.com/m8nFnpu.gif)|![](https://i.imgur.com/U3ZfKZm.gif)|![](https://i.imgur.com/dL1QD5q.gif)|
|3|![](https://i.imgur.com/M18bWwR.gif)|![](https://i.imgur.com/06HX0dS.gif)|![](https://i.imgur.com/YEBxeEm.gif)|![](https://i.imgur.com/qhmO53Q.gif)|
|4|![](https://i.imgur.com/pi2KaLC.gif)|![](https://i.imgur.com/FbbDYQg.gif)|![](https://i.imgur.com/blDy3Xh.gif)|![](https://i.imgur.com/ttYSyxm.gif)|
|5|![](https://i.imgur.com/tGHW2fY.gif)|![](https://i.imgur.com/u2IqWFy.gif)|![](https://i.imgur.com/aimCg6T.gif)|![](https://i.imgur.com/lLuBkqc.gif)|
|6|![](https://i.imgur.com/DEc16RK.gif)|![](https://i.imgur.com/tDDyco7.gif)|![](https://i.imgur.com/jpfEAVp.gif)|![](https://i.imgur.com/4xIzkkx.gif)|
|7|![](https://i.imgur.com/yIY25q7.gif)|![](https://i.imgur.com/rKpCytY.gif)|![](https://i.imgur.com/her7FH6.gif)|![](https://i.imgur.com/LwJNheJ.gif)|
|8|![](https://i.imgur.com/RE4kPBb.gif)|![](https://i.imgur.com/iEL0g3m.gif)|![](https://i.imgur.com/ACQsf8G.gif)|![](https://i.imgur.com/IIZEyxi.gif)|

主觀上，雖然相同 dimension 不同畫家產生線性變化不同，調整 style code 時，其中一個畫家產生的圖片具有另外兩個畫家畫作的特徵，有可能是因為三者皆為印象派畫風。

### 3. Mix datasets to enhance the variance of style code
---
經過前一部分的實驗，我們猜測某種畫家的畫風，會被 encode 在某個 style dimension，故該部分我們嘗試將三種不同派別畫家(or畫風)的畫混合起來做 training ，看是否能學到更多元的 style code 。

其中所選擇的三位畫家(畫風)是：浮世繪(ukiyoe)、莫內(Monet)、保羅尚賽(cezan)的畫。

這裡有個背景知識：浮世繪的油畫風格，有深刻影響到印象派(莫內)發展，而後更是受立體主義影響，從印象派演變至後印象派(保羅尚賽)。因此我們假設這三種畫風其實存在 content code(演變過程保留的feature)，而其有別於其他畫派的特色可用不同的 style code 描述。

其中以下的測試資料當中，梵谷(Vango·新印象派)的畫(如下圖b.)是沒有訓練過的，但仍然能夠產生類似的畫風。且相較於直接用梵谷的畫作 train 出來的效果要好。
###### 備註：混合三種畫的model train到110000 iteration，純粹用梵谷的畫train的model則是到120000 iteration

#### a.Photo2Monet
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/NuFTDkZ.jpg)|![](https://i.imgur.com/lVyF84M.jpg)|![](https://i.imgur.com/VqONRxd.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/4olP9ia.jpg)|![](https://i.imgur.com/te2PLHK.jpg)|![](https://i.imgur.com/6Ggznrn.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/9hCBO18.jpg)|![](https://i.imgur.com/HHRsorN.jpg)|![](https://i.imgur.com/xImDNmX.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/lKl9bxj.jpg)|![](https://i.imgur.com/8yQWqmp.jpg)|![](https://i.imgur.com/UpjW2HP.jpg)|


#### b.Photo2Vangogh
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/effAhcg.jpg)|![](https://i.imgur.com/GJJVWgi.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/rzqyM5N.jpg)|![](https://i.imgur.com/4vFWXyY.jpg)|![](https://i.imgur.com/LbMhpxW.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/qCmCXgr.jpg)|![](https://i.imgur.com/ItUGOuX.jpg)|![](https://i.imgur.com/CqktXTA.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/eBmzQ1B.jpg)|![](https://i.imgur.com/yLkzMyB.jpg)|![](https://i.imgur.com/7PDx1Cx.jpg)|

#### c.Photo2Cezanne 
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/KdgdtLu.jpg)|![](https://i.imgur.com/MYFTUBs.jpg)|![](https://i.imgur.com/YQ8HW1G.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/a1FNcI6.jpg)|![](https://i.imgur.com/mAIJef2.jpg)|![](https://i.imgur.com/tpfILqy.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/OkjezqA.jpg)|![](https://i.imgur.com/vuGsLKC.jpg)|![](https://i.imgur.com/PeMTz4s.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/07kH3NT.jpg)|![](https://i.imgur.com/LRCnx5I.jpg)|![](https://i.imgur.com/JaB8iFU.jpg)|



## Other Methods
由於上一次[作業1](https://github.com/AllenChen0958/homework1-color-transfer/blob/master/result.md)已經對其中幾個方法做過介紹，這次僅就 A Neural Algorithm of Artistic Style 以及 Diverse image2image tranfer 做介紹，剩下以轉換完之結果圖片及表格進行比較。
### [A Neural Algorithm of Artistic Style](https://ithelp.ithome.com.tw/articles/10192738)
---
採用的pre-trained的VGG的模型改版，原版VGG模型如下圖:
![](https://i.imgur.com/PYusQK8.jpg)

一般的VGG包含 convolution + pooling + FC(fully-connected layers)，但是此版本只使用16層 convolutional layers 和 5層 pooling layers，用 GAP(global average pooling) 取代 FC layers，僅保留抽取特徵的功能。

此外，模型主要包含兩個部分，Content Reconstruction 以及 Style Reconstruction。透過保留 high-layer content 融合運用 multilayer feature correlations 重建的 style 得以產生真實照片轉換成畫作的視覺效果。

#### Example
為方便閱讀比較，該方法的測資上，選擇和上方 MUNIT 三張畫作作為 style 參考，並依照 MUNIT 的三張真實照片分別就不同 style 進行轉換，該方法會隨其 pre-trained vgg-model 的好壞
#### a.Photo2Monet
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/NuFTDkZ.jpg)|![](https://i.imgur.com/lVyF84M.jpg)|![](https://i.imgur.com/VqONRxd.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/QlV20SQ.jpg)|![](https://i.imgur.com/T7Bq1GP.jpg)|![](https://i.imgur.com/If3sje2.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/hbUuDCs.jpg)|![](https://i.imgur.com/1fVXnRC.jpg)|![](https://i.imgur.com/zeUqxnU.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/gFlea6o.jpg)|![](https://i.imgur.com/dFUFiDa.jpg)|![](https://i.imgur.com/cCPkNXT.jpg)|


#### b.Photo2Vangogh
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/effAhcg.jpg)|![](https://i.imgur.com/GJJVWgi.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/SesPf3s.jpg)|![](https://i.imgur.com/nVRGcgY.jpg)|![](https://i.imgur.com/5ConY02.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/edJEfHT.jpg)|![](https://i.imgur.com/24KxhfV.jpg)|![](https://i.imgur.com/ENMI9Z7.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/2BLWcPQ.jpg)|![](https://i.imgur.com/ziAuKE1.jpg)|![](https://i.imgur.com/ki6YZgR.jpg)|

#### c.Photo2Cezanne 
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/KdgdtLu.jpg)|![](https://i.imgur.com/MYFTUBs.jpg)|![](https://i.imgur.com/YQ8HW1G.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/erEpnYK.jpg)|![](https://i.imgur.com/Dqze08V.jpg)|![](https://i.imgur.com/1Sb0hXJ.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/5qKREXR.jpg)|![](https://i.imgur.com/leMdjAi.jpg)|![](https://i.imgur.com/DQOQNCW.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/vc878Be.jpg)|![](https://i.imgur.com/voLlx5q.jpg)|![](https://i.imgur.com/tI9AMlh.jpg)|
###  [Diverse image2image tranfer](http://vllab.ucmerced.edu/hylee/publication/ECCV18_DRIT.pdf)
---
其實 DRIT 跟 MUNIT 幾乎是相同的 model ，都是分出內容空間和屬性空間(或稱風格空間)，硬要區分不同大概只有在如何融合內容空間和屬性空間上， MUNIT 使用[AdaIN](https://github.com/xunhuang1995/AdaIN-style)， DRIT 則有兩種選擇: For color-variation translate 使用 simple concatenation ； For shape-variation 使用 element-wise feature transformation 。

#### Example


#### a.Photo2Vangogh
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/effAhcg.jpg)|![](https://i.imgur.com/GJJVWgi.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/7qoN4cJ.png)|![](https://i.imgur.com/MHYPawa.png)|![](https://i.imgur.com/OzXzEDF.png)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/PywQWZo.png)|![](https://i.imgur.com/OziEJAl.png)|![](https://i.imgur.com/oBsAhmG.png)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/7VVmzF3.png)|![](https://i.imgur.com/NhEhUwB.png)|![](https://i.imgur.com/GFJPPXI.png)|

### Comparisons
#### 思路比較
[![](https://i.imgur.com/JVUDIMH.png)](https://pythonawesome.com/simple-tensorflow-implementation-of-diverse-image-to-image-translation/)
#### MUNIT vs DRIT
除了部分連接方式以及 content 和 style 結合方式的差異，其他部分十分相似。

- MUNIT:
![](https://i.imgur.com/oIgzc06.png)


- DRIT:
![](https://i.imgur.com/ZowTwSB.png)


#### Form contains differnt methods (photo2Vangogh)
| _content_  | _style_  |  Munit |Cycle GAN|fast transfer | Neural Algorith|Diverse Img2Img Transfer| 
|:-----:|:----:|:----:|:----:|:----:|:----:|:-----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/oIEitHE.jpg)|![](https://i.imgur.com/p0ZSDXs.png)|![](https://i.imgur.com/DmjDwm0.png)|![](https://i.imgur.com/SesPf3s.jpg)|![](https://i.imgur.com/80zeWHn.png)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/R79DEUA.jpg)|![](https://i.imgur.com/ipJIWGZ.png)|![](https://i.imgur.com/cnjugPS.png)|![](https://i.imgur.com/SesPf3s.png)|![](https://i.imgur.com/5zNg0Zu.png)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/L6QT7OJ.jpg)|![](https://i.imgur.com/amORqBD.png)|![](https://i.imgur.com/iJndGNW.png)|![](https://i.imgur.com/2BLWcPQ.png)|![](https://i.imgur.com/JBkYbY6.png)|




#### Features comparison
||  Munit |Cycle GAN|fast transfer |  Neural Algorithm|Diverse Img2Img Transfer| 
|:-----:|:----:|:----:|:----:|:----:|:----:|
|輸入的img類型|content_img + style_img|content_img|content_img + style_img|content_img + style_img|content_img + style_img|
|輸出的img可變性|o|x|o|o|o|
|需要training|o|o|x|x|o|
|轉換效果|筆觸+顏色|筆觸+顏色|顏色|筆觸+顏色|筆觸+顏色|
|轉換速度排名|3rd|2nd|1th|5th|4th|

## Conclusion

- MUNIT 跟 CNN 有類似的照片輸入模式，乍看之下似乎 CNN 效果比較好，然而事實上不盡然如此，如果看 CNN 的 realphoto2cezan 結果，我們會發現照片內的船幾乎消失不見，在細節保留上是比 MUNIT 差很多的，原因推測可能是因為 MUNIT 在抽取 content 特徵時，使用了 residual block，因此得以保留更多細節。相對而言，因為細節保留太多常常在 style 轉換後，仍然看起來不像畫作。

- Cycle GAN 在細節保留上，似乎取得更好的平衡，因為對其 a2b 後，尚須進行 b2a 還原測試，大部分影像轉換後的結果，仍然可以看到大部分原始照片的細節，並且畫風中的筆觸跟色彩皆有融合於其中。
- DRIT 的轉換結果在這次實驗中不是很好，原本預測應該會有與MUNIT接近的結果，但是不管是細節或是畫風轉換都沒有很出色，問題可能是出在訓練時間不夠長。


## Reference
1. https://3c.ltn.com.tw/news/35026
2. https://ithelp.ithome.com.tw/articles/10192738
3. http://vllab.ucmerced.edu/hylee/publication/ECCV18_DRIT.pdf
4. https://pythonawesome.com/simple-tensorflow-implementation-of-diverse-image-to-image-translation/
