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

class TextLineMetadata {
    public Pattern searchable;
    public float YValueMidPoint = -1;
    public float RectHeight = -1;

    public String adjacent;

    public TextLineMetadata(String regex, int patternFlags) {
        searchable = Pattern.compile(regex, patternFlags);
    }

    public void Reset() {
        YValueMidPoint = -1;
        RectHeight = -1;
    }
}

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

    private String mmddyyyy = "(.*?)(\\d{1,2}/\\d{1,2}/\\d{4})(.*)";
    private Pattern mmddyyyyPattern = Pattern.compile(mmddyyyy, Pattern.DOTALL);

    private String abbrevMonth = "((?:Jan)|(?:Feb)|(?:Mar)|(?:Apr)|(?:May)|(?:Jun)|(?:Jul)|(?:Aug)|(?:Sep)|(?:Oct)|(?:Nov)|(?:Dec))(.*?)(\\d{1,2})(.*)(\\d{4})";
    private Pattern abbrevMonthPattern = Pattern.compile(abbrevMonth, Pattern.DOTALL);
    
    private float NearnessThreshold = 3;
    private TextLineMetadata total;
    private TextLineMetadata pst;
    private TextLineMetadata gst;

    private final Map<String, String> DATE_FORMAT_REGEXPS = new HashMap<String, String>() {{
        put("^\\d{8}$", "yyyyMMdd");
        put("^\\d{1,2}-\\d{1,2}-\\d{4}$", "dd-MM-yyyy");
        put("^\\d{4}-\\d{1,2}-\\d{1,2}$", "yyyy-MM-dd");
        put(mmddyyyy, "MM/dd/yyyy");
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

        total = new TextLineMetadata("^TOTAL", Pattern.MULTILINE | Pattern.CASE_INSENSITIVE);
        gst = new TextLineMetadata("5(\\.(0*))?%", 0);
        pst = new TextLineMetadata("7(\\.(0*))?%", 0);
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
            if (total.searchable.matcher(tmpBlock.getText()).lookingAt()) {
                GraphicOverlay.Graphic blockGraphic = new TextGraphicBlock(graphicOverlay, tmpBlock);
                graphicOverlay.add(blockGraphic);
                List<FirebaseVisionText.Line> lines = tmpBlock.getLines();
                for (int j = 0; j < lines.size(); j++) {
                    if (total.searchable.matcher(lines.get(j).getText()).lookingAt()) {
                        FirebaseVisionText.Line line = lines.get(j);
                        ExtractAdjacent(line, total);
                        GraphicOverlay.Graphic lineGraphic = new TextGraphicLine(graphicOverlay, line);
                        graphicOverlay.add(lineGraphic);

                        j = lines.size();
                    }
                }
            }
            else if (abbrevMonthPattern.matcher(tmpBlock.getText()).lookingAt()) {
                GraphicOverlay.Graphic blockGraphic = new TextGraphicBlock(graphicOverlay, tmpBlock);
                graphicOverlay.add(blockGraphic);

                processDateAbbrevMonth(tmpBlock);
            }
            else if (tmpBlock.getText().contains("/")) {
                GraphicOverlay.Graphic blockGraphic = new TextGraphicBlock(graphicOverlay, tmpBlock);
                graphicOverlay.add(blockGraphic);

                Matcher match = mmddyyyyPattern.matcher(tmpBlock.getText());

                if (match.matches()) {
                    String str = match.group(2);
                    SimpleDateFormat outputFormat = new SimpleDateFormat("MM/dd/yyyy");
                    try {
                        Date date = outputFormat.parse(str);
                        outputMap.get("Date").setText(outputFormat.format(date));
                    } catch (ParseException parseX) {}
                }
            }
            else if (tmpBlock.getText().contains("%")) {
                GraphicOverlay.Graphic blockGraphic = new TextGraphicBlock(graphicOverlay, tmpBlock);
                graphicOverlay.add(blockGraphic);
                for (FirebaseVisionText.Line line : tmpBlock.getLines()) {
                    GraphicOverlay.Graphic lineGraphic = new TextGraphicLine(graphicOverlay, line);
                    graphicOverlay.add(lineGraphic);
                    if (pst.searchable.matcher(line.getText()).lookingAt()) {

                        ExtractAdjacent(line, pst);
                    }
                    if (gst.searchable.matcher(line.getText()).lookingAt()) {
//                        GraphicOverlay.Graphic lineGraphic = new TextGraphicLine(graphicOverlay, line);
//                        graphicOverlay.add(lineGraphic);
                        ExtractAdjacent(line, gst);
                    }
                }
            }
            else {
                for (FirebaseVisionText.Line line : tmpBlock.getLines()) {
                    if (Pattern.compile("\\d{1,2}(.*)\\d{1,2}").matcher(line.getText()).lookingAt()) {
                        float lineMidPoint = (line.getBoundingBox().top + line.getBoundingBox().bottom) / 2f;
                        if (Math.abs(lineMidPoint - total.YValueMidPoint) < NearnessThreshold) {
                            GraphicOverlay.Graphic lineGraphic = new TextGraphicLine(graphicOverlay, line);
                            graphicOverlay.add(lineGraphic);
                            processText(line.getText());
                        }
                        else if (Math.abs(lineMidPoint - pst.YValueMidPoint) < NearnessThreshold) {
                            GraphicOverlay.Graphic lineGraphic = new TextGraphicLine(graphicOverlay, line);
                            graphicOverlay.add(lineGraphic);
                        }
                        else if (Math.abs(lineMidPoint - gst.YValueMidPoint) < NearnessThreshold) {
                            GraphicOverlay.Graphic lineGraphic = new TextGraphicLine(graphicOverlay, line);
                            graphicOverlay.add(lineGraphic);
                        }
//                    else if (Pattern.compile("\\d{1,2}(.*)\\d{1,2}").matcher(line.getText()).lookingAt()) {
//                        GraphicOverlay.Graphic lineGraphic = new TextGraphicLine(graphicOverlay, line);
//                        graphicOverlay.add(lineGraphic);
//                    }
                    }
                }
            }
        }

        graphicOverlay.postInvalidate();
        total.Reset();
        gst.Reset();
        pst.Reset();
    }

    private void ExtractAdjacent(FirebaseVisionText.Line line, TextLineMetadata metadata) {
        metadata.RectHeight = line.getBoundingBox().top - line.getBoundingBox().bottom;
        metadata.YValueMidPoint = (line.getBoundingBox().top + line.getBoundingBox().bottom) / 2f;
    }

    private void processDateAbbrevMonth(FirebaseVisionText.TextBlock tmpBlock) {
        Matcher dateMatcher = abbrevMonthPattern.matcher(tmpBlock.getText());
        if (dateMatcher.lookingAt()) {
            String monthStr = dateMatcher.group(1);
            String dayStr = dateMatcher.group(3);
            String yearStr = dateMatcher.group(5);

            int month = abbrevMonths.get(monthStr);
            int day = Integer.parseInt(dayStr);
            int year = Integer.parseInt(yearStr);

            SimpleDateFormat outputFormat = new SimpleDateFormat("MM/dd/yyyy");
            Date date;
            try {
                date = outputFormat.parse(String.format("%d/%d/%d", month, day, year));
            } catch (ParseException parseX) {
                Calendar cal = Calendar.getInstance();
                cal.set(Calendar.YEAR, year);
                cal.set(Calendar.MONTH, month - 1);
                cal.set(Calendar.DAY_OF_MONTH, day);
                date = cal.getTime();
            }

            outputMap.get("Date").setText(outputFormat.format(date));
        }
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
