// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.firebase.samples.apps.mlkit.java.textrecognition;

import android.graphics.Bitmap;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import android.util.Log;
import android.widget.TextView;

import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;
import com.google.firebase.samples.apps.mlkit.common.CameraImageGraphic;
import com.google.firebase.samples.apps.mlkit.common.FrameMetadata;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;
import com.google.firebase.samples.apps.mlkit.java.VisionProcessorBase;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Processor for the text recognition demo.
 */
public class TextRecognitionProcessor extends VisionProcessorBase<FirebaseVisionText> {

    private static final String TAG = "TextRecProc";

    private final FirebaseVisionTextRecognizer detector;

    private Map<String, TextView> outputMap;

    public TextRecognitionProcessor(Map<String, TextView> textDict) {
        detector = FirebaseVision.getInstance().getOnDeviceTextRecognizer();
        outputMap = textDict;
    }

    @Override
    public void stop() {
        try {
            detector.close();
        } catch (IOException e) {
            Log.e(TAG, "Exception thrown while trying to close Text Detector: " + e);
        }
    }

    @Override
    protected Task<FirebaseVisionText> detectInImage(FirebaseVisionImage image) {
        return detector.processImage(image);
    }

    private float TotalYValueMidPoint = -1;
    private float TotalRectHeight = -1;
    private float NearnessThreshold = 3;

    @Override
    protected void onSuccess(
            @Nullable Bitmap originalCameraImage,
            @NonNull FirebaseVisionText results,
            @NonNull FrameMetadata frameMetadata,
            @NonNull GraphicOverlay graphicOverlay) {
        graphicOverlay.clear();
        if (originalCameraImage != null) {
            CameraImageGraphic imageGraphic = new CameraImageGraphic(graphicOverlay,
                    originalCameraImage);
            graphicOverlay.add(imageGraphic);
        }

        FirebaseVisionText.TextBlock tmpBlock;
        List<FirebaseVisionText.TextBlock> blocks = results.getTextBlocks();
        for (int i = 0; i < blocks.size(); i++) {
            tmpBlock = blocks.get(i);
            if (tmpBlock.getText().contains("TOTAL")) {
                GraphicOverlay.Graphic blockGraphic = new TextGraphicBlock(graphicOverlay, tmpBlock);
                graphicOverlay.add(blockGraphic);
                List<FirebaseVisionText.Line> lines = blocks.get(i).getLines();
                for (int j = 0; j < lines.size(); j++) {
                    if (lines.get(j).getText().contains("TOTAL")) {
                        FirebaseVisionText.Line line = lines.get(j);
                        TotalRectHeight = line.getBoundingBox().top - line.getBoundingBox().bottom;
                        TotalYValueMidPoint = (line.getBoundingBox().top + line.getBoundingBox().bottom) / 2f;
                        GraphicOverlay.Graphic lineGraphic = new TextGraphicLine(graphicOverlay, lines.get(j));
                        graphicOverlay.add(lineGraphic);
//                        lineGraphic = new TextGraphicLine(graphicOverlay, lines.get(j));
//                        graphicOverlay.add(lineGraphic);
                        j = lines.size();
                    }
                }
            }
            else {
                for (FirebaseVisionText.Line line : tmpBlock.getLines()) {
                    float lineMidPoint = (line.getBoundingBox().top + line.getBoundingBox().bottom) / 2f;
                    if (Math.abs(lineMidPoint - TotalYValueMidPoint) < NearnessThreshold) {
                        GraphicOverlay.Graphic lineGraphic = new TextGraphicLine(graphicOverlay, line);
                        graphicOverlay.add(lineGraphic);
                        if (line.getText().contains("$")){
                            outputMap.get("TOTAL").setText(line.getText());
                        }
                        else{
                            outputMap.get("TOTAL").setText("$" + line.getText());
                        }
                    }
                }
            }
        }

        graphicOverlay.postInvalidate();
        TotalYValueMidPoint = -1;
        TotalRectHeight = -1;
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.w(TAG, "Text detection failed." + e);
    }
}
