; OCR Studio — Inno Setup installer script
; Builds a Windows installer from the PyInstaller --onedir output at dist/OCRStudio/

#define MyAppName "OCR Studio"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "OCR Studio"
#define MyAppURL "https://github.com/flowa7021-source/ocr-tool"
#define MyAppExeName "OCRStudio.exe"

[Setup]
AppId={{B2F5C1A9-3C0D-4F7A-9B4C-3D81F2AA7A10}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=auto
OutputDir=.\Output
OutputBaseFilename=OCRStudio-Setup-{#MyAppVersion}
; lzma2/max (64 MB dictionary) instead of lzma2/ultra64 (1 GB dict).
; ultra64 is effectively single-threaded — Inno Setup's docs:
; "multi-threading is generally much less effective with lzma2/ultra64".
; On the ~2 GB HTR bundle that's 15-20 min single-threaded vs 5-8 min
; multi-threaded with /max on a 4-core Windows runner. Installer grows
; by ~5-10% (50-150 MB on the HTR build), an acceptable trade for
; cutting CI time by 10+ minutes. App functionality is identical —
; compression level only affects the outer container.
Compression=lzma2/max
SolidCompression=yes
; Use every available core. Default is auto, but make it explicit so
; future Inno Setup versions don't silently regress to 1 thread.
LZMANumBlockThreads=4
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0.19041
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
UninstallDisplayIcon={app}\{#MyAppExeName}
LicenseFile=..\LICENSE

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associatepdf"; Description: "Ассоциировать файлы .pdf с {#MyAppName}"; GroupDescription: "Ассоциация файлов:"; Flags: unchecked

; Opt-in wipe of user data on uninstall. Off by default so a normal
; uninstall preserves settings + downloaded GOT-OCR 2.0 weights (up to
; ~580 MB), letting the user reinstall without re-downloading.
[Code]
var
  CleanUserDataCheckbox: TNewCheckBox;

procedure InitializeUninstallProgressForm();
begin
  CleanUserDataCheckbox := TNewCheckBox.Create(UninstallProgressForm);
  CleanUserDataCheckbox.Parent := UninstallProgressForm.InnerPage;
  // Anchor the checkbox below the "Uninstalling..." status label. The
  // uninstall form does NOT have an InfoAfterLabel (that's a wizard-form
  // property); StatusLabel is always present on the uninstall form.
  CleanUserDataCheckbox.Left := UninstallProgressForm.StatusLabel.Left;
  CleanUserDataCheckbox.Top := UninstallProgressForm.StatusLabel.Top
    + UninstallProgressForm.StatusLabel.Height + ScaleY(24);
  CleanUserDataCheckbox.Width := UninstallProgressForm.InnerPage.Width
    - (CleanUserDataCheckbox.Left * 2);
  CleanUserDataCheckbox.Caption := 'Удалить все пользовательские данные (профили, логи, модели ~580 МБ)';
  CleanUserDataCheckbox.Checked := False;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  UserDataDir: String;
begin
  if (CurUninstallStep = usPostUninstall) and
     Assigned(CleanUserDataCheckbox) and CleanUserDataCheckbox.Checked then
  begin
    UserDataDir := ExpandConstant('{localappdata}\OCRStudio');
    if DirExists(UserDataDir) then
      DelTree(UserDataDir, True, True, True);
  end;
end;

[Files]
; Copy everything from PyInstaller output EXCEPT the pre-dense neural-
; network weights. On the HTR bundle those weights are ~640 MB out of
; ~2 GB total, and their entropy is so high that LZMA2 max gains 5-10%
; for 5-10 minutes of CPU. Storing them raw (``nocompression``) slashes
; the Inno Setup compile step by that same 5-10 min while inflating the
; final installer by only ~30 MB (2-3% of total). Installed size is
; unchanged — these files are copied raw to disk either way.
Source: "..\dist\OCRStudio\*"; DestDir: "{app}"; \
    Excludes: "*.safetensors,*.traineddata"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

; GOT-OCR 2.0 weights (model.safetensors, ~580 MB) — raw float tensors,
; near-incompressible. ``recursesubdirs`` preserves the original nested
; path (_internal\resources\models\got_ocr2\...) under {app}.
Source: "..\dist\OCRStudio\*.safetensors"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs nocompression

; Tesseract LSTM traineddata (rus/eng/osd, ~60 MB total) — same rationale.
Source: "..\dist\OCRStudio\*.traineddata"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs nocompression

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Optional PDF association
Root: HKCU; Subkey: "Software\Classes\.pdf\OpenWithProgids"; ValueType: string; ValueName: "OCRStudio.PDF"; ValueData: ""; Flags: uninsdeletevalue; Tasks: associatepdf
Root: HKCU; Subkey: "Software\Classes\OCRStudio.PDF"; ValueType: string; ValueName: ""; ValueData: "PDF Document"; Flags: uninsdeletekey; Tasks: associatepdf
Root: HKCU; Subkey: "Software\Classes\OCRStudio.PDF\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"; Tasks: associatepdf
Root: HKCU; Subkey: "Software\Classes\OCRStudio.PDF\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: associatepdf

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\OCRStudio\temp"
Type: filesandordirs; Name: "{localappdata}\OCRStudio\logs"
