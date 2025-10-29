import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
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
		documentSelector: [{ scheme: 'file', language: 'lust' }],
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

	const candidatePaths: string[] = [];
	if (configuredPath) {
		candidatePaths.push(configuredPath);
	}

	const exeName = process.platform === 'win32' ? 'lust-analyzer.exe' : 'lust-analyzer';
	const debugPath = context.asAbsolutePath(path.join('..', '..', 'target', 'debug', exeName));
	const releasePath = context.asAbsolutePath(path.join('..', '..', 'target', 'release', exeName));
	candidatePaths.push(debugPath, releasePath);

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

	void vscode.window.showErrorMessage(
		`Could not find lust-analyzer binary. Build the project (cargo build -p lust-analyzer) ` +
			`or set "lustAnalyzer.serverPath" to the executable.`,
	);

	return undefined;
}
