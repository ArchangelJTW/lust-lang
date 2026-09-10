import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import {
	LanguageClient,
	LanguageClientOptions,
	ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export async function activate(context: vscode.ExtensionContext) {
	const serverCommand = resolveAnalyzerPath(context);
	if (!serverCommand) {
		return;
	}

	const serverOptions: ServerOptions = {
		run: { command: serverCommand },
		debug: { command: serverCommand, args: ['--log', 'debug'] },
	};

	const clientOptions: LanguageClientOptions = {
		documentSelector: [
			{ scheme: 'file', language: 'lust' },
			{ scheme: 'untitled', language: 'lust' },
		],
		synchronize: {
			fileEvents: vscode.workspace.createFileSystemWatcher('**/*.lust'),
		},
	};

	client = new LanguageClient(
		'lustAnalyzer',
		'Lust Analyzer',
		serverOptions,
		clientOptions,
	);

	try {
		await client.start();
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		void vscode.window.showErrorMessage(
			`Failed to start lust-analyzer: ${message}`,
		);
	}
}

export async function deactivate(): Promise<void> {
	if (client) {
		await client.stop();
		client = undefined;
	}
}

function resolveAnalyzerPath(context: vscode.ExtensionContext): string | undefined {
	const config = vscode.workspace.getConfiguration('lustAnalyzer');
	const configuredPath = config.get<string>('serverPath')?.trim();

	const exeName = process.platform === 'win32' ? 'lust-analyzer.exe' : 'lust-analyzer';

	if (configuredPath) {
		if (fs.existsSync(configuredPath)) {
			return configuredPath;
		}
		const foundConfiguredInPath = findInPath(configuredPath);
		if (foundConfiguredInPath) {
			return foundConfiguredInPath;
		}
	}

	const candidatePaths: string[] = [];

	// Check open workspace folders (target/release then target/debug)
	const workspaceFolders = vscode.workspace.workspaceFolders ?? [];
	for (const folder of workspaceFolders) {
		candidatePaths.push(
			path.join(folder.uri.fsPath, 'target', 'release', exeName),
			path.join(folder.uri.fsPath, 'target', 'debug', exeName),
		);
	}

	// Check relative to extension location (target/release then target/debug)
	candidatePaths.push(
		context.asAbsolutePath(path.join('..', 'target', 'release', exeName)),
		context.asAbsolutePath(path.join('..', 'target', 'debug', exeName)),
		context.asAbsolutePath(path.join('..', '..', 'target', 'release', exeName)),
		context.asAbsolutePath(path.join('..', '..', 'target', 'debug', exeName)),
	);

	// Check ~/.cargo/bin
	const cargoBinPath = path.join(os.homedir(), '.cargo', 'bin', exeName);
	candidatePaths.push(cargoBinPath);

	const resolved = candidatePaths.find((candidate) => {
		if (!candidate) {
			return false;
		}
		try {
			return fs.existsSync(candidate);
		} catch {
			return false;
		}
	});

	if (resolved) {
		return resolved;
	}

	// Check system PATH
	const pathBinary = findInPath(exeName);
	if (pathBinary) {
		return pathBinary;
	}

	void vscode.window.showErrorMessage(
		`Could not find lust-analyzer binary. Build the project (cargo build -p lust-analyzer) ` +
			`or set "lustAnalyzer.serverPath" to the executable.`,
	);

	return undefined;
}

function findInPath(exeName: string): string | undefined {
	const envPath = process.env.PATH || '';
	const delimiter = path.delimiter;
	for (const dir of envPath.split(delimiter)) {
		if (!dir) {
			continue;
		}
		const fullPath = path.join(dir, exeName);
		try {
			if (fs.existsSync(fullPath)) {
				return fullPath;
			}
		} catch {
			// ignore access errors
		}
	}
	return undefined;
}
