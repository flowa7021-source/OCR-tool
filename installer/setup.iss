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
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0.19041
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
UninstallDisplayIcon={app}\{#MyAppExeName}
LicenseFile=..\LICENSE
SetupLogging=yes

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associatepdf"; Description: "Ассоциировать файлы .pdf с {#MyAppName}"; GroupDescription: "Ассоциация файлов:"; Flags: unchecked

[Files]
; Copy everything from PyInstaller output
Source: "..\dist\OCRStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

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
