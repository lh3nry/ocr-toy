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
package com.google.firebase.samples.apps.mlkit.java;

import android.app.Dialog;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.app.ActivityCompat.OnRequestPermissionsResultCallback;
import androidx.core.content.ContextCompat;

import com.google.android.gms.common.annotation.KeepName;
import com.google.android.material.textfield.TextInputLayout;
import com.google.firebase.ml.vision.objects.FirebaseVisionObjectDetectorOptions;
import com.google.firebase.samples.apps.mlkit.R;
import com.google.firebase.samples.apps.mlkit.common.CameraSource;
import com.google.firebase.samples.apps.mlkit.common.CameraSourcePreview;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;
import com.google.firebase.samples.apps.mlkit.java.automl.AutoMLImageLabelerProcessor;
import com.google.firebase.samples.apps.mlkit.java.barcodescanning.BarcodeScanningProcessor;
import com.google.firebase.samples.apps.mlkit.java.custommodel.CustomImageClassifierProcessor;
import com.google.firebase.samples.apps.mlkit.java.facedetection.FaceContourDetectorProcessor;
import com.google.firebase.samples.apps.mlkit.java.facedetection.FaceDetectionProcessor;
import com.google.firebase.samples.apps.mlkit.java.imagelabeling.ImageLabelingProcessor;
import com.google.firebase.samples.apps.mlkit.java.objectdetection.ObjectDetectorProcessor;
import com.google.firebase.samples.apps.mlkit.java.textrecognition.TextRecognitionProcessor;

import org.jetbrains.annotations.NotNull;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * Demo app showing the various features of ML Kit for Firebase. This class is used to
 * set up continuous frame processing on frames from a camera source.
 */
@KeepName
public final class LivePreviewActivity extends AppCompatActivity
        implements OnRequestPermissionsResultCallback {
    private static final String TAG = "LivePreviewActivity";
    private static final int PERMISSION_REQUESTS = 1;

    private CameraSource cameraSource = null;
    private CameraSourcePreview preview;
    private GraphicOverlay graphicOverlay;

    private Button vendorNameButton;
    private Button saveButton;

    File targetDir;

    private Dialog vendorDialog;
    private LinearLayout vendorList;

    private Dialog newVendorDialog;
    private TextInputLayout newVendorInput;

    private Map<String, TextView> textDict = new HashMap<>();

    private final String OUTPUT_DIR_NAME = "/OCRCSV";
    private final String OUTPUT_FILE_COMMON_NAME = "/Export.csv";
    private TextRecognitionProcessor textRecognitionProcessor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");
        setContentView(R.layout.activity_live_preview);

        targetDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + OUTPUT_DIR_NAME);
        if (!targetDir.exists()){
            targetDir.mkdirs();
        }

        preview = findViewById(R.id.firePreview);
        if (preview == null) {
            Log.d(TAG, "Preview is null");
        }
        graphicOverlay = findViewById(R.id.fireFaceOverlay);
        if (graphicOverlay == null) {
            Log.d(TAG, "graphicOverlay is null");
        }

        LinearLayout entriesLayout = findViewById(R.id.EntriesLayout);
        CreateEntry("TOTAL", "$0.00", entriesLayout);
        CreateEntry("GST", "$0.00", entriesLayout);
        CreateEntry("PST", "$0.00", entriesLayout);
        CreateEntry("Date", "??", entriesLayout);

        saveButton = findViewById(R.id.saveButton);
        saveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                OnSaveTapped();
            }
        });
        saveButton.setVisibility(View.GONE);

        vendorNameButton = findViewById(R.id.vendorNameButton);
        vendorNameButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (vendorDialog == null) {
                     SetupVendorDialog();
                }
                vendorDialog.show();
            }
        });

        if (allPermissionsGranted()) {
            createCameraSource();
        } else {
            getRuntimePermissions();
        }
    }

    private void OnSaveTapped() {
        if (vendorNameButton.getText() == "?") {
            // Shouldn't be able to press before assigning a vendor due to visibility.
            throw new IllegalStateException("Vendor name unknown.");
        }

        String csvPath = GetCurrentPath(vendorNameButton.getText().toString()) + OUTPUT_FILE_COMMON_NAME;
        File exported = new File(csvPath);
        if (!exported.exists()){
            WriteHeader(GetCurrentPath(vendorNameButton.getText().toString()) + OUTPUT_FILE_COMMON_NAME,  textDict.keySet().toArray(new String[textDict.size()]));
        }

        try {
            FileReader reader = new FileReader(exported);
            BufferedReader buffReader = new BufferedReader(reader);
            String headers = buffReader.readLine();
            buffReader.close();
            reader.close();

            StringTokenizer tokenizer = new StringTokenizer(headers, ",");
            int numColumns = tokenizer.countTokens();

            if (numColumns != textDict.size()) {
                throw new IllegalStateException(
                        "There is a mismatch in the number of header columns ("
                                + numColumns
                                +") and number of entries ("
                                + textDict.size()
                                +") in the TextView dictionary");
            }

            FileWriter writer = new FileWriter(exported, true);
            writer.append(ConstructCSVLine(tokenizer));

            writer.close();

            textRecognitionProcessor.needToClearData = true;

        } catch (Exception e) {

        }
    }

    private String ConstructCSVLine(StringTokenizer tokenizer)
    {
        StringBuilder sb = new StringBuilder();
        String token;
        while (tokenizer.hasMoreTokens()) {
            token = tokenizer.nextToken();
            if (textDict.containsKey(token)) {
                sb.append(textDict.get(token).getText());
                sb.append(",");
            }
            else {
                sb.append("***,");
            }
        }
        sb.append("\n");

        return sb.toString();
    }

    private void WriteHeader(final String filename, final String[] headers) {
        new Thread() {
            public void run () {

                try {
                    FileWriter writer = new FileWriter(filename);
                    for (String header : headers) {
                        writer.append(header);
                        writer.append(",");
                    }

                    writer.append("\n");
                    writer.close();
                } catch (Exception e) {

                }
            }
        }.start();
    }

    private void SetupVendorDialog(){
        vendorDialog = new Dialog(LivePreviewActivity.this);
        vendorDialog.setContentView(R.layout.dialog_layout);
        vendorDialog.setTitle("Set Vendor");

        vendorList = vendorDialog.findViewById(R.id.vendorButtonList);

        for (File f : targetDir.listFiles()) {
            if (f.isDirectory()) {
                AddVendorButton(f.getName());
            }
        }

        Button newVendorButton = vendorDialog.findViewById(R.id.newVendorButton);
        newVendorButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (newVendorDialog == null) {
                    SetupNewVendorDialog();
                }
                else {
                    if (newVendorInput == null) {
                        newVendorInput = newVendorDialog.findViewById(R.id.vendorNameInput);
                    }
                    newVendorInput.getEditText().getText().clear();
                }

                newVendorDialog.show();
            }
        });
    }

    @NotNull
    private String GetCurrentPath(String vendorName) {
        return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                + OUTPUT_DIR_NAME
                + "/" + vendorName;
    }

    private void SetupNewVendorDialog() {
        newVendorDialog = new Dialog(LivePreviewActivity.this);
        newVendorDialog.setContentView(R.layout.new_vendor_dialog);
        newVendorDialog.setTitle("New Vendor");

        if (newVendorInput == null) {
            newVendorInput = newVendorDialog.findViewById(R.id.vendorNameInput);
        }

        Button addButton = newVendorDialog.findViewById(R.id.addNewVendorButton);
        addButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String vendorName = newVendorInput.getEditText().getText().toString();
                if (vendorName == null || vendorName.isEmpty()){
                    newVendorInput.getEditText().setHint("You must set a vendor name!");
                    newVendorInput.getEditText().setHintTextColor(Color.RED);
                    return;
                }

                String dirPath = GetCurrentPath(vendorName);
                File newVendorDir = new File(dirPath);
                if (!newVendorDir.exists()){
                    newVendorDir.mkdirs();
                    WriteHeader(dirPath + OUTPUT_FILE_COMMON_NAME, textDict.keySet().toArray(new String[textDict.size()]));
                }

                AddVendorButton(vendorName);
                newVendorDialog.dismiss();
                vendorNameButton.setText(vendorName);
                vendorDialog.dismiss();
                saveButton.setVisibility(View.VISIBLE);
                textRecognitionProcessor.needToClearData = true;
            }
        });
    }

    private void AddVendorButton(final String buttonText) {
        if (vendorDialog == null) {
            throw new IllegalStateException("We do not have a reference to the vendor selection dialog!");
        }

        Button newButton = new Button(this);
        newButton.setText(buttonText);
        newButton.setTextSize(20f);
        newButton.setAllCaps(false);
        newButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                VendorSelectionCallback(buttonText);
            }
        });

        vendorList.addView(newButton);
    }

    private void VendorSelectionCallback(String vendorName)
    {
        vendorNameButton.setText(vendorName);
        vendorDialog.dismiss();
        saveButton.setVisibility(View.VISIBLE);
        textRecognitionProcessor.needToClearData = true;
    }

    private void CreateEntry(String label, String defaultValue, LinearLayout parent)
    {
        LinearLayout entryHLayout = new LinearLayout(this);
        entryHLayout.setOrientation(LinearLayout.HORIZONTAL);

        // label TextView
        TextView labelText = new TextView(this);
        labelText.setTextColor(Color.WHITE);
        labelText.setTextSize(36f);
        labelText.setText(label);
        labelText.setLayoutParams(
            new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
            ));

        // dollar amount TextView
        TextView valueText = new TextView(this);
        valueText.setTextColor(Color.WHITE);
        valueText.setTextSize(36f);
        valueText.setText(defaultValue);
        valueText.setTextAlignment(View.TEXT_ALIGNMENT_GRAVITY);
        valueText.setGravity(Gravity.RIGHT);
        valueText.setLayoutParams(
            new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
            ));

        entryHLayout.addView(labelText);
        entryHLayout.addView(valueText);

        parent.addView(entryHLayout);

        textDict.put(label, valueText);
    }


    private void createCameraSource() {
        // If there's no existing cameraSource, create one.
        if (cameraSource == null) {
            cameraSource = new CameraSource(this, graphicOverlay);
        }

        try {
            textRecognitionProcessor = new TextRecognitionProcessor(textDict);
            cameraSource.setMachineLearningFrameProcessor(textRecognitionProcessor);
        } catch (Exception e) {
            Log.e(TAG, "Can not create image processor", e);
            Toast.makeText(
                    getApplicationContext(),
                    "Can not create image processor: " + e.getMessage(),
                    Toast.LENGTH_LONG)
                    .show();
        }
    }

    /**
     * Starts or restarts the camera source, if it exists. If the camera source doesn't exist yet
     * (e.g., because onResume was called before the camera source was created), this will be called
     * again when the camera source is created.
     */
    private void startCameraSource() {
        if (cameraSource != null) {
            try {
                if (preview == null) {
                    Log.d(TAG, "resume: Preview is null");
                }
                if (graphicOverlay == null) {
                    Log.d(TAG, "resume: graphOverlay is null");
                }
                preview.start(cameraSource, graphicOverlay);
            } catch (IOException e) {
                Log.e(TAG, "Unable to start camera source.", e);
                cameraSource.release();
                cameraSource = null;
            }
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");
        startCameraSource();
    }

    /**
     * Stops the camera.
     */
    @Override
    protected void onPause() {
        super.onPause();
        preview.stop();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraSource != null) {
            cameraSource.release();
        }
    }

    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    this.getPackageManager()
                            .getPackageInfo(this.getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                return false;
            }
        }
        return true;
    }

    private void getRuntimePermissions() {
        List<String> allNeededPermissions = new ArrayList<>();
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                allNeededPermissions.add(permission);
            }
        }

        if (!allNeededPermissions.isEmpty()) {
            ActivityCompat.requestPermissions(
                    this, allNeededPermissions.toArray(new String[0]), PERMISSION_REQUESTS);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, @NonNull int[] grantResults) {
        Log.i(TAG, "Permission granted!");
        if (allPermissionsGranted()) {
            createCameraSource();
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    private static boolean isPermissionGranted(Context context, String permission) {
        if (ContextCompat.checkSelfPermission(context, permission)
                == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Permission granted: " + permission);
            return true;
        }
        Log.i(TAG, "Permission NOT granted: " + permission);
        return false;
    }
}
