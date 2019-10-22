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

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Demo app showing the various features of ML Kit for Firebase. This class is used to
 * set up continuous frame processing on frames from a camera source.
 */
@KeepName
public final class LivePreviewActivity extends AppCompatActivity
        implements OnRequestPermissionsResultCallback,
        OnItemSelectedListener,
        CompoundButton.OnCheckedChangeListener {
    private static final String FACE_DETECTION = "Face Detection";
    private static final String OBJECT_DETECTION = "Object Detection";
    private static final String AUTOML_IMAGE_LABELING = "AutoML Vision Edge";
    private static final String TEXT_DETECTION = "Text Detection";
    private static final String BARCODE_DETECTION = "Barcode Detection";
    private static final String IMAGE_LABEL_DETECTION = "Label Detection";
    private static final String CLASSIFICATION_QUANT = "Classification (quantized)";
    private static final String CLASSIFICATION_FLOAT = "Classification (float)";
    private static final String FACE_CONTOUR = "Face Contour";
    private static final String TAG = "LivePreviewActivity";
    private static final int PERMISSION_REQUESTS = 1;

    private CameraSource cameraSource = null;
    private CameraSourcePreview preview;
    private GraphicOverlay graphicOverlay;
    private String selectedModel = TEXT_DETECTION;

    private Button vendorNameButton;

    File targetDir;

    private Dialog vendorDialog;
    private LinearLayout vendorList;

    private Dialog newVendorDialog;
    private TextInputLayout newVendorInput;

    private Map<String, TextView> textDict = new HashMap<>();

    private final String OUTPUT_DIR_NAME = "/OCRCSV";
    private final String OUTPUT_FILE_COMMON_NAME = "/Export.csv";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");
        setContentView(R.layout.activity_live_preview);

        targetDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + OUTPUT_DIR_NAME);
        if (!targetDir.exists()){
            targetDir.mkdirs();
        }

        final String filename = targetDir.toString() + OUTPUT_FILE_COMMON_NAME;
        TestWrite(filename);


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
        CreateEntry("Date", "??", entriesLayout);

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
            createCameraSource(selectedModel);
        } else {
            getRuntimePermissions();
        }
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

    private void TestWrite(final String filename) {
        new Thread() {
            public void run() {
                try {

                    FileWriter fw = new FileWriter(filename, true);

                    fw.append("No");
                    fw.append(',');

                    fw.append("code");
                    fw.append(',');

                    fw.append("nr");
                    fw.append(',');

                    fw.append("Orde");
                    fw.append(',');

                    fw.append("Da");
                    fw.append(',');

                    fw.append("Date");
                    fw.append(',');

                    fw.append("Leverancier");
                    fw.append(',');

                    fw.append("Baaln");
                    fw.append(',');

                    fw.append("asd");
                    fw.append(',');

                    fw.append("Kwaliteit");
                    fw.append(',');

                    fw.append("asd");
                    fw.append(',');

                    fw.append('\n');



                    // fw.flush();
                    fw.close();

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

                String dirPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + OUTPUT_DIR_NAME + "/" + vendorName;
                File newVendorDir = new File(dirPath);
                if (!newVendorDir.exists()){
                    newVendorDir.mkdirs();
                    WriteHeader(dirPath + OUTPUT_FILE_COMMON_NAME, textDict.keySet().toArray(new String[textDict.size()]));
                }

                AddVendorButton(vendorName);
                newVendorDialog.dismiss();
                vendorNameButton.setText(vendorName);
                vendorDialog.dismiss();
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

    @Override
    public synchronized void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
        // An item was selected. You can retrieve the selected item using
        // parent.getItemAtPosition(pos)
        selectedModel = parent.getItemAtPosition(pos).toString();
        Log.d(TAG, "Selected model: " + selectedModel);
        preview.stop();
        if (allPermissionsGranted()) {
            createCameraSource(selectedModel);
            startCameraSource();
        } else {
            getRuntimePermissions();
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
        // Do nothing.
    }

    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        Log.d(TAG, "Set facing");
        if (cameraSource != null) {
            if (isChecked) {
                cameraSource.setFacing(CameraSource.CAMERA_FACING_FRONT);
            } else {
                cameraSource.setFacing(CameraSource.CAMERA_FACING_BACK);
            }
        }
        preview.stop();
        startCameraSource();
    }

    private void createCameraSource(String model) {
        // If there's no existing cameraSource, create one.
        if (cameraSource == null) {
            cameraSource = new CameraSource(this, graphicOverlay);
        }

        try {
            switch (model) {
                case CLASSIFICATION_QUANT:
                    Log.i(TAG, "Using Custom Image Classifier (quant) Processor");
                    cameraSource.setMachineLearningFrameProcessor(new CustomImageClassifierProcessor(this, true));
                    break;
                case CLASSIFICATION_FLOAT:
                    Log.i(TAG, "Using Custom Image Classifier (float) Processor");
                    cameraSource.setMachineLearningFrameProcessor(new CustomImageClassifierProcessor(this, false));
                    break;
                case TEXT_DETECTION:
                    Log.i(TAG, "Using Text Detector Processor");
                    cameraSource.setMachineLearningFrameProcessor(new TextRecognitionProcessor(textDict));
                    break;
                case FACE_DETECTION:
                    Log.i(TAG, "Using Face Detector Processor");
                    cameraSource.setMachineLearningFrameProcessor(new FaceDetectionProcessor(getResources()));
                    break;
                case AUTOML_IMAGE_LABELING:
                    cameraSource.setMachineLearningFrameProcessor(new AutoMLImageLabelerProcessor(this));
                    break;
                case OBJECT_DETECTION:
                    Log.i(TAG, "Using Object Detector Processor");
                    FirebaseVisionObjectDetectorOptions objectDetectorOptions =
                            new FirebaseVisionObjectDetectorOptions.Builder()
                                    .setDetectorMode(FirebaseVisionObjectDetectorOptions.STREAM_MODE)
                                    .enableClassification().build();
                    cameraSource.setMachineLearningFrameProcessor(
                            new ObjectDetectorProcessor(objectDetectorOptions));
                    break;
                case BARCODE_DETECTION:
                    Log.i(TAG, "Using Barcode Detector Processor");
                    cameraSource.setMachineLearningFrameProcessor(new BarcodeScanningProcessor());
                    break;
                case IMAGE_LABEL_DETECTION:
                    Log.i(TAG, "Using Image Label Detector Processor");
                    cameraSource.setMachineLearningFrameProcessor(new ImageLabelingProcessor());
                    break;
                case FACE_CONTOUR:
                    Log.i(TAG, "Using Face Contour Detector Processor");
                    cameraSource.setMachineLearningFrameProcessor(new FaceContourDetectorProcessor());
                    break;
                default:
                    Log.e(TAG, "Unknown model: " + model);
            }
        } catch (Exception e) {
            Log.e(TAG, "Can not create image processor: " + model, e);
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
            createCameraSource(selectedModel);
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
