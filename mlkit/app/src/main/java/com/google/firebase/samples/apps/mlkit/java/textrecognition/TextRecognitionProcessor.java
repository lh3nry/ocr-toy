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
import java.text.DecimalFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Processor for the text recognition demo.
 */
public class TextRecognitionProcessor extends VisionProcessorBase<FirebaseVisionText> {

    private static final String TAG = "TextRecProc";

    private final FirebaseVisionTextRecognizer detector;

    private Map<String, TextView> outputMap;
    private Map<String, Map<Float, Integer>> counterSet = new HashMap<>();

    private final Map<String, Integer> abbrevMonths = new HashMap<String, Integer> () {{
        put("Jan",  1);
        put("Feb",  2);
        put("Mar",  3);
        put("Apr",  4);
        put("May",  5);
        put("Jun",  6);
        put("Jul",  7);
        put("Aug",  8);
        put("Sep",  9);
        put("Oct", 10);
        put("Nov", 11);
        put("Dec", 12);
    }};

    Pattern checkMonths = Pattern.compile("^(Jan)|(Feb)|(Mar)|(Apr)|(May)|(Jun)|(Jul)|(Aug)|(Sep)|(Oct)|(Nov)|(Dec)*$");

    String mmddyyyy_str = "^\\d{1,2}/\\d{1,2}/\\d{4}";
    Pattern mmddyyyy = Pattern.compile(mmddyyyy_str);

    Pattern totalPattern = Pattern.compile("^TOTAL");

    private final Map<String, String> DATE_FORMAT_REGEXPS = new HashMap<String, String>() {{
        put("^\\d{8}$", "yyyyMMdd");
        put("^\\d{1,2}-\\d{1,2}-\\d{4}$", "dd-MM-yyyy");
        put("^\\d{4}-\\d{1,2}-\\d{1,2}$", "yyyy-MM-dd");
        put(mmddyyyy_str, "MM/dd/yyyy");
        put("^\\d{4}/\\d{1,2}/\\d{1,2}$", "yyyy/MM/dd");
        put("^\\d{1,2}\\s[a-z]{3}\\s\\d{4}$", "dd MMM yyyy");
        put("^\\d{1,2}\\s[a-z]{4,}\\s\\d{4}$", "dd MMMM yyyy");
        put("^\\d{12}$", "yyyyMMddHHmm");
        put("^\\d{8}\\s\\d{4}$", "yyyyMMdd HHmm");
        put("^\\d{1,2}-\\d{1,2}-\\d{4}\\s\\d{1,2}:\\d{2}$", "dd-MM-yyyy HH:mm");
        put("^\\d{4}-\\d{1,2}-\\d{1,2}\\s\\d{1,2}:\\d{2}$", "yyyy-MM-dd HH:mm");
        put("^\\d{1,2}/\\d{1,2}/\\d{4}\\s\\d{1,2}:\\d{2}$", "MM/dd/yyyy HH:mm");
        put("^\\d{4}/\\d{1,2}/\\d{1,2}\\s\\d{1,2}:\\d{2}$", "yyyy/MM/dd HH:mm");
        put("^\\d{1,2}\\s[a-z]{3}\\s\\d{4}\\s\\d{1,2}:\\d{2}$", "dd MMM yyyy HH:mm");
        put("^\\d{1,2}\\s[a-z]{4,}\\s\\d{4}\\s\\d{1,2}:\\d{2}$", "dd MMMM yyyy HH:mm");
        put("^\\d{14}$", "yyyyMMddHHmmss");
        put("^\\d{8}\\s\\d{6}$", "yyyyMMdd HHmmss");
        put("^\\d{1,2}-\\d{1,2}-\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$", "dd-MM-yyyy HH:mm:ss");
        put("^\\d{4}-\\d{1,2}-\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}$", "yyyy-MM-dd HH:mm:ss");
        put("^\\d{1,2}/\\d{1,2}/\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$", "MM/dd/yyyy HH:mm:ss");
        put("^\\d{4}/\\d{1,2}/\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}$", "yyyy/MM/dd HH:mm:ss");
        put("^\\d{1,2}\\s[a-z]{3}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$", "dd MMM yyyy HH:mm:ss");
        put("^\\d{1,2}\\s[a-z]{4,}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$", "dd MMMM yyyy HH:mm:ss");
    }};

    public boolean needToClearData = false;

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

        if (needToClearData) {
            counterSet.clear();
            needToClearData = false;
        }

        for (FirebaseVisionText.TextBlock tmpBlock : results.getTextBlocks()) {
            if (totalPattern.matcher(tmpBlock.getText()).lookingAt()) {
                GraphicOverlay.Graphic blockGraphic = new TextGraphicBlock(graphicOverlay, tmpBlock);
                graphicOverlay.add(blockGraphic);
                List<FirebaseVisionText.Line> lines = tmpBlock.getLines();
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
            else if (checkMonths.matcher(tmpBlock.getText()).lookingAt()) {
                GraphicOverlay.Graphic blockGraphic = new TextGraphicBlock(graphicOverlay, tmpBlock);
                graphicOverlay.add(blockGraphic);

                for ( FirebaseVisionText.Line line : tmpBlock.getLines() ) {
                    if (checkMonths.matcher(line.getText()).lookingAt()) {
                        if (line.getElements().size() > 3) {
                            processDateAbbrevMonth(line);
                        }
                        for (FirebaseVisionText.Element e : line.getElements()) {
                            GraphicOverlay.Graphic elementGraphic = new TextGraphicElement(graphicOverlay, e);
                            graphicOverlay.add(elementGraphic);
                        }
                    }
                }
            }
            else if (tmpBlock.getText().contains("/")) {
                for (FirebaseVisionText.Line line : tmpBlock.getLines()) {
                    if (mmddyyyy.matcher(line.getText()).lookingAt()) {
                        GraphicOverlay.Graphic blockGraphic = new TextGraphicLine(graphicOverlay, line);
                        graphicOverlay.add(blockGraphic);

                        Matcher match = mmddyyyy.matcher(line.getText());
                        if (match.matches()) {
                            String str = match.group();
                            try {
                                Date date = new SimpleDateFormat("MM/dd/yyyy").parse(str);
                                outputMap.get("Date").setText(date.toString());
                            } catch (ParseException parseX) {}
                        }
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
                            processText(line.getText());
                        }
                        else{
                            processText(line.getText());
                        }
                    }
                }
            }
        }

        graphicOverlay.postInvalidate();
        TotalYValueMidPoint = -1;
        TotalRectHeight = -1;
    }
    
    private void processDateAbbrevMonth(FirebaseVisionText.Line line) {
        String monthText = line.getElements().get(0).getText();

        int month = abbrevMonths.get(monthText);
        int day = Integer.parseInt(line.getElements().get(1).getText()
                .replace(",", "")
                .replace(".", ""));
        int year = Integer.parseInt(line.getElements().get(2).getText());

        Date date;
        try {
            date = new SimpleDateFormat("dd-M-yyyy").parse(String.format("%d-%d-%d", day, month, year));
        } catch (ParseException parseEx) {
            Calendar cal = Calendar.getInstance();
            cal.set(Calendar.YEAR, year);
            cal.set(Calendar.MONTH, month - 1);
            cal.set(Calendar.DAY_OF_MONTH, day);
            date = cal.getTime();
        }

        outputMap.get("Date").setText(date.toString());
    }

    private float FindMaxOccuring (Map<Float, Integer> countingMap){
        int max = -1;
        float maxKey = Float.NaN;
        for (Map.Entry<Float, Integer> entry : countingMap.entrySet()) {
            if (entry.getValue() > max){
                max = entry.getValue();
                maxKey = entry.getKey();
            }
        }

        return maxKey;
    }

    private void processText(String raw) {
        raw = raw.replace("$","");
        raw = raw.replace(" ","");
        raw = raw.replace(",",".");
        try {
            float parsed = Float.parseFloat(raw);
            if (counterSet.containsKey("TOTAL")){
                Map<Float, Integer> x = counterSet.get("TOTAL");
                if (x.containsKey(parsed)) {
                    x.put (parsed, x.get(parsed) + 1);
                }
                else {
                    x.put (parsed, 1);
                }
            }
            else {
                Map<Float, Integer> x = new HashMap<>();
                x.put(parsed, 1);
                counterSet.put("TOTAL", x);
            }

            outputMap.get("TOTAL").setText(new DecimalFormat("#.00").format(FindMaxOccuring(counterSet.get("TOTAL"))));
        } catch (NumberFormatException formatEx) {
//            outputMap.get("TOTAL").setText("$" + raw);
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.w(TAG, "Text detection failed." + e);
    }
}
